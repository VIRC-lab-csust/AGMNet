_base_ = './faster_rcnn_r50_caffe_fpn_1x_coco.py'
model = dict(
    backbone=dict(depth=152))
