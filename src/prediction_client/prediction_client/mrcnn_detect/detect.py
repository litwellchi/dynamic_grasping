from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2


class mmdetDetector(object):
    def __init__(self, config_file, checkpoint_file):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_detector(
            self.config_file, self.checkpoint_file, device='cuda:0')
        self.result = None
        self.img = None
        self.depth = None

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
        bound = [x_min, x_max, y_min, y_max]
        """
        x_min, x_max, y_min, y_max = [375, 875, 175, 525]

    def show_result(self, img, result, save_path=None):
        """
        save or show the detection result
        """
        self.model.show_result(img, result, out_file=save_path)


if __name__ == '__main__':
    # Test code
    config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
    checkpoint_file = './epoch_2_09061103.pth'
    mmdetDetector(config_file, checkpoint_file)
