import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector, init_detector

import cv2
import numpy as np
import copy
import re

import vitpose_infer
import extrapolation
import utils

def main(params):
    
    # define the detection model
    det_model = init_detector(
        params.det_config, params.det_checkpoint, device=params.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        params.pose_config, params.pose_checkpoint, device=params.device.lower())
    
    # define the directory to store the results
    vitpose_res_name = re.split('\.',params.video_path)[0].split('\\')[-1]
    res_dir=vitpose_res_name+'_results'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    # (I) 
    # Get the ViTPose Inference results
    if not os.path.exists(os.path.join(res_dir,'vitpose_kpt_'+vitpose_res_name+'.npz')):
        #infer keypoints from ViTPose
        kseq, size, fps = vitpose_infer.get_kpt_seq(params,params.video_path,det_model,pose_model)
        
        # save the ViTPose results for future use
        np.savez(os.path.join(res_dir,'vitpose_kpt_'+vitpose_res_name),kseq=kseq, size=size, fps=fps)
    
    # or just load
    else:
        load_tmp=np.load(os.path.join(res_dir,'vitpose_kpt_'+vitpose_res_name+'.npz'))
        kseq=load_tmp['kseq']
        size=load_tmp['size']
        fps=load_tmp['fps']
    
    # (II)  
    # infer the occluded keypoints
    inferred_kpt_seq,missing_kpt = extrapolation.infer_missing_kpt(params,kseq,params.lpf_cutoff,fps,size)

    # save the inferred_kpt_seq
    np.savez(os.path.join(res_dir,'inferred_kpt_'+vitpose_res_name),inferred_kpt_seq=inferred_kpt_seq,missing_kpt=missing_kpt)
    
    
    # (III) 
    #  plot the results
    utils.view_missing_kpt(len(kseq),missing_kpt,out_path=os.path.join(res_dir,vitpose_res_name+'_occluded_pt'))
    
    # video with ViTPose result
    utils.put_video_file(params,pose_model,os.path.join(res_dir,'ViTPose_'+vitpose_res_name+'.mp4'),kseq)
    utils.put_video_file(params,pose_model,os.path.join(res_dir,'polyfit+lpf_'+vitpose_res_name+'.mp4'),inferred_kpt_seq)
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--video_path', help='path of the video file')
    parser.add_argument('--det_config', help='path of config file for detection')
    parser.add_argument('--det_checkpoint', help='path of checkpoint file for detection')
    parser.add_argument('--pose_config', help='path of config file for pose')
    parser.add_argument('--pose_checkpoint', help='path of checkpoint file for pose')
    parser.add_argument('--lpf_cutoff', default=2, help='cutoff frequency of the low pass filter')
    parser.add_argument('--device', default='cuda:0', help='device for inference')
    args = parser.parse_args()
    
    params = utils.Params(args.video_path,args.det_config,args.det_checkpoint,
    args.pose_config,args.pose_checkpoint,args.lpf_cutoff,args.device)
    main(params)