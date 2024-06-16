#Adapted by Srijeet Roy from: https://github.com/amitrana001/DynaMITe/blob/main/train_net.py

import numpy as np
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import itertools
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import json
import logging

from typing import Any, Dict, List, Set

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
)
from dynamite.utils.misc import default_argument_parser

# from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dynamite import (
    COCOLVISDatasetMapper, EvaluationDatasetMapper
)

from dynamite import (
    add_maskformer2_config,
    add_hrnet_config
)

from dynamite.inference.utils.eval_utils import log_single_instance, log_multi_instance
from dynamite.inference.multi_instance.random_best_worst import evaluate

from metrics.summary import summarize_results,summarize_round_results
import copy
import gc
import tarvis_dynamite_helpers as helpers
_DATASET_ROOT = helpers._DATASET_ROOT
_DATASET_PATH = helpers._DATASET_PATH

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to Mask2Former.
    """

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        mapper = EvaluationDatasetMapper(cfg,False,dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)        # d2 call
    
    @classmethod
    def interactive_evaluation(cls, cfg, dynamite_model, interactions, iou, 
                                all_images, all_gt_masks, dataloader_dict,
                                args=None, tarvis_config=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        print('[INFO] Interactive Evaluation started...')
        if not args:
            return 

        logger = logging.getLogger(__name__)

        if args and args.eval_only:
            eval_datasets = args.eval_datasets      
            vis_path = args.vis_path                
            eval_strategy = args.eval_strategy      
            seed_id = args.seed_id
            iou_threshold = args.iou_threshold
            max_interactions = args.max_interactions
            max_rounds = args.max_rounds
            save_masks = args.save_masks
            debug = args.debug
        
        if not isinstance(iou_threshold, list):                
            iou_threshold = [iou_threshold]
        if not isinstance(max_interactions, list):                
            max_interactions = [max_interactions]

        for dataset_name in eval_datasets:

            if dataset_name in ["davis_2017_val","mose_val","sbd_multi_insts","burst_val","coco_2017_val"]:
                print(f'[INFO] Initiating Multi-Instance Evaluation on {eval_datasets}...')
                if dataset_name == "davis_2017_val":
                    tarvis_config['dataset'] = 'DAVIS'                                

                save_path = os.path.join(vis_path, f'{interactions}_interactions/iou_{int(iou*100)}')
                #save_path = vis_path
                os.makedirs(save_path, exist_ok=True) 
                expt_path = os.path.join(save_path, 'tarvis_propagation')
                os.makedirs(expt_path, exist_ok=True)

                print(f'[INFO] Starting evaluation...')
                vis_path_vis = os.path.join(save_path, 'vis')
                os.makedirs(vis_path_vis, exist_ok=True)
                results_i, progress_report = evaluate(dynamite_model,
                                    tarvis_config,
                                    dataloader_dict, all_images, all_gt_masks,
                                    iou_threshold = iou,
                                    max_interactions = interactions,
                                    eval_strategy = eval_strategy,
                                    seed_id=seed_id,
                                    vis_path=vis_path_vis,
                                    max_rounds=max_rounds,
                                    dataset_name=dataset_name,
                                    save_masks=True,
                                    expt_path=expt_path)
                
                print(f'[INFO] Evaluation complete for dataset {dataset_name}: IoU threshold={iou}, Interaction budget={interactions}!')

                if dataset_name not in ["mose_val", "burst_val"]:
                    with open(os.path.join(save_path,f'results_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(results_i, f)
                    with open(os.path.join(save_path,f'progress_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(progress_report, f)
                
                    summary, df = summarize_results(results_i)
                    df.to_csv(os.path.join(save_path, f'round_results_{interactions}_interactions_iou_{int(iou*100)}.csv'))
                    with open(os.path.join(save_path,f'summary_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(summary, f)
                    
                    summary_df = summarize_round_results(df, iou)
                    summary_df.to_csv(os.path.join(save_path, f'round_summary_{interactions}_interactions_iou_{int(iou*100)}.csv'))
                del results_i,progress_report

def setup(args):
    """
    Create configs and perform basic setups.
    """
    print('[INFO] Setting up DynaMITE...')
    cfg = get_cfg()
    # for poly lr schedule
    #add_deeplab_config(cfg)
    add_maskformer2_config(cfg)                 
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)       # path to config file
    cfg.merge_from_list(args.opts)
    cfg.freeze()                                # make cfg (and children) immutable
    default_setup(cfg, args)                    # D2 call
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dynamite")
    return cfg


def main(args):
    
    cfg = setup(args)       # create configs 
    print('[INFO] Setup complete!')

    dataset_name = args.eval_datasets[0]
    all_gt_masks = {}
    all_images = {}
    if dataset_name not in ["mose_val", "burst_val"]:        
        all_images = helpers.load_images(dataset_name, args.debug)
        all_gt_masks = helpers.load_gt_masks(dataset_name, args.debug)    
        assert len(all_images) == len(all_gt_masks)
    
    print(f'[INFO] Iterating through the Data Loader...')
    if dataset_name in ["burst_val"]:
        dataloader_dict = helpers.burst_imset()
    else:
        data_loader = Trainer.build_test_loader(cfg, dataset_name)
        dataloader_dict = defaultdict(list)
        for idx, inputs in enumerate(data_loader):      
            curr_seq_name = inputs[0]["file_name"].split('/')[-2]
            if args.debug and curr_seq_name != list(all_gt_masks.keys())[0]:
                break
            dataloader_dict[curr_seq_name].append([idx, inputs])
        del data_loader

    for interactions, iou in list(itertools.product(args.max_interactions,args.iou_threshold)):
        data = copy.deepcopy(dataloader_dict)
        # for evaluation
        if args.eval_only:
            print('[INFO] DynaMITExTarViS Evaluation!')
            torch.autograd.set_grad_enabled(False)

            print('[INFO] Building model...')
            dynamite_model = Trainer.build_model(cfg)                                                # load model (torch.nn.Module)
            print('[INFO] Loading model weights...')                                        
            DetectionCheckpointer(dynamite_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(           # d2 checkpoint load
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            print('[INFO] DynaMITe loaded!')

            tarvis_config = {}
            tarvis_config['model_path'] = "/globalwork/roy/dynamite_video/tarvis_dynamite/pretrained_backbones/swin-tiny_finetune/090000.pth"
            tarvis_config['dataset'] = args.eval_datasets        
            tarvis_config['output_dir'] = ""
            tarvis_config["mask_directory"] = ""
            
            tarvis_config["overwrite"] = False
            tarvis_config["split"] = None
            tarvis_config["amp"] = True
            tarvis_config["num_workers"] = 2
            tarvis_config["viz"] = False
            tarvis_config["viz_num_procs"] = 8
            tarvis_config["seq_split"] = None
            tarvis_config["seq_names"] = None
            tarvis_config["clip_length"] = None
            tarvis_config["frame_overlap"] = None
            tarvis_config["min_dim"] = None
            tarvis_config["disable_resize"] = False
            tarvis_config["vos_bg_grid_size"] = None        
            
            res = Trainer.interactive_evaluation(cfg, dynamite_model, 
                                                interactions, iou,
                                                all_images, all_gt_masks, data,
                                                args, tarvis_config)

        #return res
        del dynamite_model, res, data
        torch.cuda.empty_cache()
        gc.collect()

    else:
        print(f'[INFO] Training routine... Not Implemented')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("[INFO] Command Line Args:", args)
    launch(                                                                            
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )