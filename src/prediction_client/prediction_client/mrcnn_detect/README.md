NOTE:
The MMDetection path in configs need to be modified.

### Set up the mmedetection environment 
Please check the torch version before install. Recommend to install the mmdetection from sourse.
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
Then change the base path in configs.
```_base_ = '{mmdetection}/configs/mask_rcnn/mask_rcnn_r18_caffe_fpn_mstrain-poly_1x_coco.py' ```
