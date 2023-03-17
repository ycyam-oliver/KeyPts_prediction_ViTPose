import os
import ffmpeg
import cv2
import numpy as np

from mmpose.apis import vis_pose_result

from mmpose.datasets import DatasetInfo

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Params():
    # create params object for other functions to use
    def __init__(self,video_path,det_config,det_checkpoint,pose_config,pose_checkpoint,lpf_cutoff,device):
        
        self.video_path=video_path
        self.det_config=det_config
        self.det_checkpoint=det_checkpoint
        self.pose_config=pose_config
        self.pose_checkpoint=pose_checkpoint
        self.lpf_cutoff=lpf_cutoff
        self.device='cuda:0'
        self.det_cat_id=1
        self.bbox_thr=0.5
        self.kpt_thr=0.35
    
def view_missing_kpt(len_kpt_seq,missing_list,out_path=None):
    # missing_list is a list of (j,i)
    # which means missing ith point in the j th frame
    stat_img=np.zeros((17,len_kpt_seq))
    frame_stat=[0]*len_kpt_seq
    for frame, pt in missing_list:
        stat_img[pt][frame]=1
        frame_stat[frame]+=1
    
    f, (a0, a1) = plt.subplots(1, 2, figsize=(8,3),width_ratios=[1, 1])
    
    a0.imshow(stat_img,interpolation='none',aspect='auto',origin='lower')
    a0.set_xlabel('frame')
    a0.set_ylabel('keypoint index')
    a0.grid(which='major', color='w', linestyle='-', linewidth=0.5)
    a0.yaxis.set_major_locator(MaxNLocator(integer=True))

    a1.plot(frame_stat,'.', markersize=0.5)
    a1.set_xlabel('frame')
    a1.set_ylabel('# missing keypoints')
    a1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path+'.pdf',bbox_inches='tight',pad_inches=0)
        
    # plt.show()
    
def put_video_file(params,pose_model,out_file,kpt_seq):

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(params.video_path)

    # save_out_video:
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(out_file, fourcc,fps, size)

    frame_idx=-1
    while (cap.isOpened()):

        frame_idx+=1

        flag, image = cap.read()
        if not flag:
            break

        annotated_img=vis_pose_result(
            pose_model,
            image,
            [{'keypoints':kpt_seq[frame_idx]}], #<--- plot this!
            kpt_score_thr=params.kpt_thr,
            dataset=dataset,
            dataset_info=dataset_info)

        #save_out_video:
        videoWriter.write(annotated_img)

    cap.release()
    videoWriter.release()