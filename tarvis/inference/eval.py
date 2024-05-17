import sys
sys.path.append("/globalwork/roy/dynamite_video/tarvis_dynamite/TarViS_DynaMITe")
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, Any
from tarvis.utils.timer import Timer

from tarvis.utils.paths import Paths
from tarvis.inference.tarvis_inference_model import TarvisInferenceModel, AllObjectsLostException
from tarvis.config import cfg
from tarvis.data.inference_dataset import InferenceDataset as SequenceClipGenerator
from tarvis.data.collate import collate_fn_inference
from tarvis.inference.result_accumulators import (
    InstanceSegmentationResultAccumulator,
    ExemplarBasedSegmentationResultAccumulator,
    PanopticSegmentationResultAccumulator
)
from tarvis.inference.visualization import save_vizualization

import tarvis.inference.result_formatters as rf
import tarvis.inference.dataset_parser as parsers

import os
os.environ['TARVIS_WORKSPACE_DIR'] = "/globalwork/roy/dynamite_video/tarvis_dynamite/"
import os.path as osp
import torch
import yaml
import shutil


@Timer.log_duration("inference")
def process_sequence(model: TarvisInferenceModel, mixed_precision: bool, data_loader: DataLoader,
                     sequence_info: Dict[str, Any], inference_params: Dict[str, Any]):

    if model.task_type == "panoptic_seg":
        result_accumulator = PanopticSegmentationResultAccumulator(
            sequence_length=len(sequence_info["image_paths"]),
            max_tracks_per_clip=inference_params["MAX_TRACKS_PER_CLIP"],
            track_score_threshold=inference_params["TRACK_SCORE_THRESHOLD"],
            original_image_size=sequence_info["image_dims"],
            image_padding=None,
            thing_class_ids=sequence_info["thing_class_ids"],
            tensor_ops_on_cpu=inference_params.get("POSTPROCESS_ON_CPU", False)
        )

    elif model.task_type == "instance_seg":
        result_accumulator = InstanceSegmentationResultAccumulator(
            sequence_length=len(sequence_info["image_paths"]),
            max_tracks_per_clip=inference_params["MAX_TRACKS_PER_CLIP"],
            track_score_threshold=inference_params["TRACK_SCORE_THRESHOLD"],
            produce_semseg_output=False,
            original_image_size=sequence_info["image_dims"],
            image_padding=None
        )

    elif model.task_type == "vos":
        result_accumulator = ExemplarBasedSegmentationResultAccumulator(
            sequence_length=len(sequence_info["image_paths"]),
            first_ref_mask_frame_index=sequence_info["first_ref_mask_frame_index"],
            original_image_size=sequence_info["image_dims"],
            image_padding=None
        )

    else:
        raise ValueError("Should not be here")

    model.reset_vos_buffers()

    for i, clip_data in enumerate(tqdm(data_loader, leave=False, desc="Clip")):
        if i == 0:
            img_height, img_width = clip_data["images"].shape[-2:]  # [H, W]
            padded_height, padded_width = model.compute_padded_dims(img_height, img_width)
            result_accumulator.image_padding = padded_width - img_width, padded_height - img_height  # [pad_right, pad_bottom]

        images = clip_data["images"].cuda()
        try:
            with autocast(enabled=mixed_precision):
                model_outputs = model(
                    images=images, clip_frame_indices=clip_data["frame_indices"],
                    vos_ref_mask_info=clip_data["vos_ref_mask_info"]
                )


        except AllObjectsLostException as _:
            result_accumulator.previous_clip_to_rles()
            result_accumulator.reset_buffers()
            model.reset_vos_buffers()
            continue
        
        torch.cuda.empty_cache()
        result_accumulator.add_clip_result(
            model_output=model_outputs, frame_indices=clip_data["frame_indices"]
        )
    return result_accumulator.finalize_output()


def dict_to_namespace(d):
    return type('', (), d)

@torch.no_grad()
def eval_tarvis(base_args):
    print(f'Eval TarViS')
    if isinstance(base_args, dict):
        args = dict_to_namespace(base_args)
    else:
        args = base_args

    if not osp.isabs(args.model_path):
        args.model_path = osp.join(Paths.saved_models_dir(), args.model_path)

    expected_cfg_path = osp.join(osp.dirname(args.model_path), "config.yaml")
    assert osp.exists(expected_cfg_path), f"Config file not found at expected path: {expected_cfg_path}"
    cfg.merge_from_file(expected_cfg_path)

    cfg_dataset = cfg.DATASETS.get(args.dataset).INFERENCE.as_dict()

    image_resize_params = cfg_dataset["IMAGE_RESIZE"]
    assert not (args.min_dim and args.disable_resize)

    if args.clip_length:
        cfg_dataset["CLIP_LENGTH"] = args.clip_length

    if args.frame_overlap:
        cfg_dataset["FRAME_OVERLAP"] = args.frame_overlap

    if args.min_dim:
        image_resize_params["MODE"] = "min_dim"
        image_resize_params["MIN_DIMS"] = args.min_dim

    elif args.disable_resize:
        image_resize_params["MODE"] = "none"

    model = TarvisInferenceModel(args.dataset).cuda()
    model.restore_weights(args.model_path)
    model = model.eval()

    if args.vos_bg_grid_size:
        model.vos_query_extractor.bg_grid_size[0] = args.vos_bg_grid_size[0]
        model.vos_query_extractor.bg_grid_size[1] = args.vos_bg_grid_size[1]

    # hard-set output directory
    if args.output_dir:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")

    output_results_dir = output_dir

    if args.dataset == "YOUTUBE_VIS":
        dataset_info = parsers.YoutubeVISParser(**Paths.youtube_vis_val_paths())
        result_formatter = rf.OVISResultFormatter(output_results_dir, max_tracks_per_video=10)

    elif args.dataset == "OVIS":
        dataset_info = parsers.OVISParser(**Paths.ovis_val_paths())
        result_formatter = rf.OVISResultFormatter(output_results_dir, max_tracks_per_video=20)

    elif args.dataset == "DAVIS":
        if cfg_dataset["SPLIT"] == "val":
            # dataset_info = parsers.DavisDatasetParser(**Paths.davis_val_paths())
            dataset_info = parsers.DavisDatasetParser(images_base_dir='/globalwork/roy/dynamite_video/tarvis_dynamite/dataset_images/inference/davis', 
                                                      annotations_base_dir=args.mask_directory,
                                                      image_set_file_path='/globalwork/roy/dynamite_video/tarvis_dynamite/dataset_annotations/inference/davis/ImageSet_val.txt',
                                                      seq=args.seq_names)
        else:
            assert cfg_dataset["SPLIT"] == "testdev", f"Invalid split: {cfg_dataset['SPLIT']}"
            dataset_info = parsers.DavisDatasetParser(**Paths.davis_testdev_paths())

        result_formatter = rf.DavisResultFormatter(output_results_dir, split=cfg_dataset["SPLIT"])

    elif args.dataset == "BURST":
        if cfg_dataset["SPLIT"] == "val":
            paths = Paths.burst_val_anns()
        else:
            paths = Paths.burst_test_anns()
        dataset_info = parsers.BurstDatasetParser(**paths,
                                                  mode=cfg_dataset["MODE"],
                                                  num_interleaved_frames=10)
        result_formatter = rf.BURSTResultFormatter(output_results_dir, dataset_info.first_frame_annotations_file,
                                                   seq_split_spec=args.seq_split)

    elif args.dataset == "KITTI_STEP":
        dataset_info = parsers.KittiStepParser(**Paths.kitti_step_val_paths())
        result_formatter = rf.KITTISTEPResultFormatter(
            output_results_dir, track_score_threshold=cfg_dataset["TRACK_SCORE_THRESHOLD"]
        )

    elif args.dataset == "CITYSCAPES_VPS":
        dataset_info = parsers.CityscapesVPSParser(
            **Paths.cityscapes_vps_val_paths(), sparse_frames=cfg_dataset["SPARSE_FRAMES"]
        )
        result_formatter = rf.CityscapesVPSResultFormatter(
            output_results_dir,  track_score_threshold=cfg_dataset["TRACK_SCORE_THRESHOLD"]
        )

    elif args.dataset == "VIPSEG":
        if cfg_dataset["SPLIT"] == "val":
            dataset_info = parsers.VIPSegDatasetParser(**Paths.vipseg_val_paths())
        elif cfg_dataset["SPLIT"] == "test":
            dataset_info = parsers.VIPSegDatasetParser(**Paths.vipseg_test_paths())
        else:
            raise ValueError(f"Invalid split specified: {cfg_dataset['SPLIT']}")

        result_formatter = rf.VIPSegResultFormatter(
            output_results_dir, track_score_threshold=cfg_dataset["TRACK_SCORE_THRESHOLD"]
        )

    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    dataset_info.partition_sequences(args.seq_split)

    if args.seq_names:
        dataset_info.isolate_sequences(args.seq_names)

    total_frames_processed = 0
    pbar = tqdm(dataset_info, leave=False)

    for i, sequence_info in enumerate(pbar):
        pbar.set_description(f"{sequence_info['dirname']}")

        seq_clip_generator = SequenceClipGenerator(
            task_type=dataset_info.task_type,
            image_paths=sequence_info["image_paths"],
            clip_length=cfg_dataset['CLIP_LENGTH'],
            overlap_length=cfg_dataset['FRAME_OVERLAP'],
            image_resize_params=image_resize_params,
            first_frame_mask_paths=sequence_info.get("first_frame_mask_paths", None),
            first_frame_mask_rles=sequence_info.get("first_frame_mask_rles", None),
            first_frame_object_points=sequence_info.get("first_frame_object_points", None),
            first_ref_mask_frame_index=sequence_info.get("first_ref_mask_frame_index", 0)
        )

        data_loader = DataLoader(
            seq_clip_generator, shuffle=False, batch_size=1, num_workers=args.num_workers,
            collate_fn=collate_fn_inference
        )

        try:
            sequence_results = process_sequence(
                model=model,
                mixed_precision=args.amp,
                data_loader=data_loader,
                sequence_info=sequence_info,
                inference_params=cfg_dataset
            )
            sequence_results = result_formatter.add_sequence_result(
                accumulator_output=sequence_results,
                sequence_info=sequence_info
            )

        except Exception as exc:
            tqdm.write(f"Error occurred while processing sequence {sequence_info['dirname']} at index {i}.")
            raise exc

        total_frames_processed += len(sequence_info["image_paths"])

        if args.viz:
            seq_vis_dir = osp.join(output_dir, "vizualization", sequence_info["dirname"])
            save_vizualization(
                task_type=dataset_info.task_type,
                output_dir=seq_vis_dir,
                sequence_info=sequence_info,
                sequence_results=sequence_results,
                category_labels=dataset_info.category_labels,
                num_processes=args.viz_num_procs
            )

    # some datasets write out a single file combining all the per-sequence results
    result_formatter.finalize_output()

    # print(f"Inference duration: {Timer.get_duration('inference')}")
    # print(f"Inference FPS: {float(total_frames_processed) / Timer.get_duration('inference')}")
    # if Timer.exists("vizualization"):
    #     print(f"Visualization duration: {Timer.get_duration('vizualization')}")
    
    return sequence_results["mask_tensors"]


# if __name__ == "__main__":
#     parser = ArgumentParser()

#     parser.add_argument(
#         "--expt", type=str, required=True, help='name of the experiment, determines mask and output directories'
#     )
#     args = parser.parse_args()

#     #mask_directory = os.path.join("/globalwork/roy/dynamite_video/tarvis_dynamite/dataset_annotations/inference/davis", args.expt)
#     mask_directory = "/globalwork/roy/dynamite_video/tarvis_dynamite/dataset_annotations/inference/davis/DynaMITe_3_85"
#     output_dir = os.path.join("/globalwork/roy/dynamite_video/tarvis_dynamite/output/", args.expt)
    
#     if os.path.isdir(output_dir):
#         print(f'An experiment with the same name {args.expt} exists at {output_dir}')
#         resp = input('Do you want to overwrite? [y/n]:')
#         if resp == 'y':
#             shutil.rmtree(output_dir)
#             os.makedirs(output_dir)
#     else:
#         os.makedirs(output_dir)

    
#     setattr(args, "model_path", "/globalwork/roy/dynamite_video/tarvis_dynamite/pretrained_backbones/swin-tiny_finetune/090000.pth")
#     setattr(args, "dataset", "DAVIS")
#     setattr(args, "output_dir", output_dir)
#     setattr(args, "overwrite", False)
#     setattr(args, "split", None)
#     setattr(args, "amp", True)
#     setattr(args, "num_workers", 2)
#     setattr(args, "viz", False)
#     setattr(args, "viz_num_procs", 8)
#     setattr(args, "seq_split", None)
#     setattr(args, "seq_names", ['blackswan'])
#     setattr(args, "clip_length", None)
#     setattr(args, "frame_overlap", None)
#     setattr(args, "min_dim", None)
#     setattr(args, "disable_resize", False)
#     setattr(args, "vos_bg_grid_size", None)
#     setattr(args, "mask_directory", mask_directory)
#     eval_tarvis(args)
    
