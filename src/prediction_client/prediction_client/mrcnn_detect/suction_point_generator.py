import cv2
from detect import mmdetDetector

class SuctionPointGenerator(object):
    def __init__(self,config_file, checkpoint_file):
        self.rgb_detector=mmdetDetector(config_file, checkpoint_file)


    def find_suction_point(self, img, depth, show_result=False):
        """
        Use this function to get the suction point
        """
        result = self.rgb_detector.instance_detect(img)

        # if show_result: self.show_result(img,result)

    def get_scution_candidate(self, masks, method='simple_midpoint'):
        """
        return:pose_list:[[x,y,z,r,p,y],...]
        """
        return getattr(self, '__'+method)(masks)
    
if __name__ == '__main__':
    # Test code
    config_file = '/home/xwchi/autostore_franka/dynamic_grasping/src/prediction_client/prediction_client/mrcnn_detect/configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
    checkpoint_file = '/home/xjgao/InstanceSeg_opaque/exps_opaque/epoch_2_09061103.pth'
    detector = mmdetDetector(config_file, checkpoint_file)
    img = cv2.imread('/home/xwchi/autostore_franka/dynamic_grasping/src/prediction_client/prediction_client/collect_data/color_image_1.png')
    x_min, x_max, y_min, y_max = [375, 875, 175, 525]
    # crop_img = img[y_min:y_max, x_min:x_max]
    crop_img = img
    result = detector.instance_detect(crop_img)
    detector.show_result(crop_img,result,'/home/xwchi/autostore_franka/dynamic_grasping/src/prediction_client/prediction_client/mrcnn_detect/test.jpg')
