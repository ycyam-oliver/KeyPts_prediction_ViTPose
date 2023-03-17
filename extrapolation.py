import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import signal
import copy



def infer_missing_kpt(params,kpt_seq,freq_cutoff,fps,size):

    # use polyfit to infer missing point
    # and use low pass filter to maintain the overall smoothness in time
    
    num_prev_frame = 5 #number of previous frames used for polyfit
    
    #====================================================================
    # define a low pass filter for the keypoints
    nyq = 0.5 * fps # Nyquist freq
    order = 2 # polynomial order
    normal_cutoff = freq_cutoff / nyq
    
    # filter parameters:
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)  
    z_orig = signal.lfilter_zi(b, a)

    num_kps = 17
    
    # z is for tracking all the keypoints
    x_track = [z_orig for _ in range(num_kps)] 
    y_track = [z_orig for _ in range(num_kps)]

    def filter(x, z):
        x_, z = signal.lfilter(b, a, [x], zi=z)
        return  x_, z

    def preprocess_kp(kp):
        for i in range(num_kps):
            # x coor
            temp, x_track[i] = filter(kp[i][0], x_track[i])
            kp[i][0] = temp

            # y coor
            temp, y_track[i] = filter(kp[i][1], y_track[i])
            kp[i][1] = temp
        return kp
    
    #====================================================================
    
    kpt_seq=copy.deepcopy(kpt_seq)
    missing_kpt=[]
    
    # for different frames
    for j in range(len(kpt_seq)):
        
        kpt=copy.deepcopy(kpt_seq[j])
        
        # loop over the kepoints
        for i in range(num_kps):
                
            if kpt[i][2]<params.kpt_thr: #i.e. if it is a missing point
                
                # if it is inside the image
                if 0<=kpt[i][0]<size[0] and 0<=kpt[i][1]<size[1]: 
                    missing_kpt.append((j,i)) #ith point in the j th frame
                    
                    
                # interpolate forward

                times=[]
                xs=[]
                ys=[]
                left_ptr=j
                
                for _ in range(num_prev_frame):
                    left_ptr-=1
                    if left_ptr<=0: break
                    kpt_now=copy.deepcopy(kpt_seq[left_ptr][i])
                    if kpt_now[2]>params.kpt_thr:
                        times.append(left_ptr)
                        xs.append(kpt_now[0])
                        ys.append(kpt_now[1])
                        
                if len(times)>=3:

                    z1=np.polyfit(times,xs,3)
                    p1=np.poly1d(z1)

                    z2=np.polyfit(times,ys,3)
                    p2=np.poly1d(z2)

                    if 0<=p1(j)<size[0] and 0<=p2(j)<size[1]:
                        kpt[i][2]=10 # use 10 to mark the edited kpoint inside the image
                        
                    kpt[i][0]=p1(j)
                    kpt[i][1]=p2(j)
        
        # pass to the low pass filter
        if j==0:
            for ind in range(num_kps):
                x_track[ind]=kpt_seq[j][ind][0]*x_track[ind]
                y_track[ind]=kpt_seq[j][ind][1]*y_track[ind]

        else:
            kpt_seq[j]=preprocess_kp(kpt) 
                        
    missing_kpt.sort()
    return kpt_seq, missing_kpt







