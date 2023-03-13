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
    config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
    checkpoint_file = './epoch_2_09061103.pth'
    spg = SuctionPointGenerator(config_file, checkpoint_file)