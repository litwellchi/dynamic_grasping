from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2


config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
time_start = time.time()
checkpoint_name = 'epoch_2'
# checkpoint_file  = 'exps_opaque_mix_data/' + checkpoint_name +'.pth'
checkpoint_file  = '/home/xjgao/InstanceSeg_opaque/exps_opaque/epoch_2_09061103.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("Time consumed model load: ", time.time()-time_start)