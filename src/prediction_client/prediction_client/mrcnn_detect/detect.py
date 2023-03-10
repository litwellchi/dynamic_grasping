from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2


class SuctionPointDetector(object):
    def __init__(self, config_file, checkpoint_file):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_detector(
            self.config_file, self.checkpoint_file, device='cuda:0')
        self.result = None
        self.img = None
        self.depth = None

    def find_suction_point(self, img, depth, show_result=False):
        """
        Use this function to get the suction point
        """
        result = self.instance_detect(img)
        # masks =
        suction_candidate = self.get_scution_candidate(masks)

        # if show_result: self.show_result(img,result)

    def instance_detect(self, img):
        """
        detect the object on RGB image
        """
        time_start = time.time()
        result = inference_detector(self.model, img)
        print("Time consumed: ", time.time()-time_start)
        return result

    def set_box_bound(self, bound):
        """
        Set the RGB range of picking box
        """
        x_min, x_max, y_min, y_max = 375, 875, 175, 525

    def get_scution_candidate(self, masks, method='simple_midpoint'):
        """
        return:pose_list:[[x,y,z,r,p,y],...]
        """
        return getattr(self, '__'+method)(masks)

    def show_result(self, img, result, save_path=None):
        """
        save or show the detection result
        """
        self.model.show_result(img, result, out_file=save_path)

    def __simple_midpoint(masks):
        print("__simple_midpoint")
        pass

    def __depth2pcd():
        pass


if __name__ == '__main__':
    # Test code
    config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
    checkpoint_file = './epoch_2_09061103.pth'
    SuctionPointDetector(config_file, checkpoint_file)
