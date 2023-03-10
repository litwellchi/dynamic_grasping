from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2


class SuctionPointDetector(object):
    def __init__(self, args):
        config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
        checkpoint_file = './epoch_2_09061103.pth'
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.result = None
        pass

    def detect(self, img):
        """
        detect the object on RGB image
        """
        result = inference_detector(model, img)
        print("Time consumed: ", time.time()-time_start)
        self.result = result
        return

    def set_box_bound(self, bound):
        """
        Set the RGB range of picking box
        """
        x_min, x_max, y_min, y_max = 375, 875, 175, 525

    def get_scution_candidate(self, method='simple_midpoint'):
        """
        return:pose_list:[[x,y,z,r,p,y],...]
        """
        pass

    def show_result(self, save_path=None):
        """
        save or show the detection result
        """
        model.show_result(self.img, self.result, out_file=save_path)
        pass

    def __simple_midpoint():
        pass

    def __depth2pcd():
        pass
