import os
import warnings

from mmpose.apis import inference_top_down_pose_model, process_mmdet_results
from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector
    
import cv2
import numpy as np
import copy

def get_kpt_seq(params,video_path,det_model,pose_model):
    # use ViTPose model to infer the keypoints of a person near the center of an image
    
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    kpt_seq = []
    
    frame_idx=-1
    while (cap.isOpened()):
        
        frame_idx+=1

        flag, image = cap.read()
        if not flag:
            break

        # using a detection model to infer the bbox of different people
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, params.det_cat_id)
        
        # only get one box for one person (by choosing the one with largest area)
        if len(person_results)>1:
            bbox_res = []
            for pi in range(len(person_results)):
                if person_results[pi]['bbox'][4]>params.bbox_thr:
                    x1,y1,x2,y2=person_results[pi]['bbox'][:4]
                    area_p=(x2-x1)*(y2-y1)
                    overlap_num=0
                    for bi in range(len(bbox_res)):
                        x1_,y1_,x2_,y2_=bbox_res[bi]['bbox'][:4]
                        area_b=(x2_-x1_)*(y2_-y1_)
                        left_x,left_y=max(x1,x1_),max(y1,y1_)
                        right_x,right_y=min(x2,x2_),min(y2,y2_)
                        overlap_area=(right_x-left_x)*(right_y-left_y)
                        if overlap_area/min(area_p,area_b)>0.5:
                            overlap_num+=1
                            if area_p>area_b: 
                                bbox_res[bi]=copy.deepcopy(person_results[pi])
                                break
                    if overlap_num==0:
                        bbox_res.append(person_results[pi])
            person_results=bbox_res
        
        # consider only the person closest to the centre of the image
        if len(person_results)>1:
            
            dist_min=float('inf')
            for b_ind in range(len(person_results)):
                
                bbox=person_results[b_ind]['bbox'][0:4]
                dist_curr=((bbox[0]+bbox[2])/2-size[0]/2)**2+\
                ((bbox[1]+bbox[3])/2-size[1]/2)**2
                if dist_curr<dist_min:
                    dist_min=dist_curr
                    curr_res=person_results[b_ind]
                        
        else:
            curr_res=person_results[0]

        # test the box with the central person

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image,
            [curr_res],
            bbox_thr=params.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info)
        
        kpt_seq.append(pose_results[0]['keypoints'])

    cap.release()
    
    # kpt_seq is an array of size 17x3 
    # size[0]=horizontal size, size[1]=vertical size
    return kpt_seq, size, fps
    
    