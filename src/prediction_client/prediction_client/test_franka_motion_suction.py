import sys

from franka_interfaces.srv import CartMotionTime
from franka_interfaces.srv import JointMotionVel
from franka_interfaces.srv import PosePath
from franka_interfaces.srv import FrankaHand
from franka_interfaces.msg import RobotState
import rclpy
from rclpy.node import Node
import transform
from srt_serial import PythonSerialDriver
from relay_control import RelaySerialDriver

import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

from mrcnn_detect.detect import mmdetDetector
import pyrealsense2 as rs
import math
import yaml
import argparse
import os
import time
import numpy as np
import sys
import cv2

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
config_file = '/home/xwchi/autostore_franka/dynamic_grasping/src/prediction_client/prediction_client/mrcnn_detect/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
checkpoint_file = '/home/xjgao/InstanceSeg_opaque/exps_opaque/epoch_2_09061103.pth'

detector = mmdetDetector(config_file, checkpoint_file)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


def detect_once(self,show_result = True, camera_crop_x_left=350,camera_crop_x_right=1000,camera_crop_y_top=100,camera_crop_y_bottom=550):
        # Wait for a coherent pair of frames: depth and color
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        while not depth_image.any() or not color_image.any():
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()

        # Convert images to numpy arrays
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        t_start = time.time()  # 开始计时

        crop_img = color_image[camera_crop_y_top:camera_crop_y_bottom+1,camera_crop_x_left:camera_crop_x_right+1,:]
        result = detector.instance_detect(crop_img)

        t_end = time.time()  # 结束计时\

        # get suction point by simple 
        camera_xyz_list = []
        for box in result[0][0]:
            
            #generate suction point by simple midpoint
            y_min,x_min,y_max,x_max,confidence=box
            center_x = int((x_min+x_max)/2) + camera_crop_x_left                 
            center_y = int((y_min+y_max)/2) + camera_crop_y_top
            # center_z = depth_image[center_y,center_x]
            dis = aligned_depth_frame.get_distance(center_x, center_y)
            camera_xyz = rs.rs2_deproject_pixel_to_point(
            depth_intrin, (center_x, center_y), dis)  # 计算相机坐标系的xyz
            camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
            camera_xyz = camera_xyz.tolist()
            if camera_xyz[0] != 0.0 and camera_xyz[1] != 0.0 and camera_xyz[2] != 0.0: #remove zero list
                camera_xyz_list.append(camera_xyz)
            if show_result:
                cv2.circle(depth_colormap, (center_y, center_x), 10, (255,255,255), 0)        
        return camera_xyz_list