from mmdet.registry import VISUALIZERS
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image
import sys

# 判断是否在 Jupyter Notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'Shell']
    except NameError:
        return False

def infer_detect_picture(model, image_file_path='demo/demo.jpg', out_file='result_demo.jpg'):
    image = mmcv.imread(image_file_path, channel_order='rgb')
    result = inference_detector(model, image)
    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 保存可视化图像到文件
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=out_file,
    )

    # 自动显示
    if is_notebook():
        # 在 notebook 中显示图像
        from IPython.display import display
        display(Image.open(out_file))
    else:
        # 在终端显示图像
        img = mmcv.imread(out_file, channel_order='rgb')
        plt.imshow(img)
        plt.axis('off')
        plt.title('Detection Result')
        plt.show()

###########################        
# 使用示例
# !mim download mmdet --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ./checkpoints
# import mmcv
# import mmengine
# from mmdet.apis import init_detector, inference_detector
# from mmdet.utils import register_all_modules
# # Choose to use a config and initialize the detector
# config_file = 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
# # Setup a checkpoint file to load
# checkpoint_file = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# # register all modules in mmdet into the registries
# register_all_modules()

# # build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# infer_detect_picture(model)