from mmdet.registry import VISUALIZERS
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image
import sys
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# Choose to use a config and initialize the detector
config_file_mask_rcnn = 'configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
config_file_sparse_rcnn = 'configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py'
# Setup a checkpoint file to load
checkpoint_file_mask_rcnn = '/mnt/ali-sh-1/usr/amiao1/wangleyang/nn_ml_mid_term_task2/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'
checkpoint_file_sparse_rcnn = '/mnt/ali-sh-1/usr/amiao1/wangleyang/nn_ml_mid_term_task2/mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'

# register all modules in mmdet into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model_mask_rcnn = init_detector(config_file_mask_rcnn, checkpoint_file_mask_rcnn, device='cpu')  # or device='cuda:0'
model_sparse_rcnn = init_detector(config_file_sparse_rcnn, checkpoint_file_sparse_rcnn, device='cpu')  # or device='cuda:0'

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

# 测试集
with open('data/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
image_list = [line.strip() for line in lines]

# 选前50张测试集
for image_idx in image_list[:50]:
    image_path = f'data/VOCdevkit/VOC2007/JPEGImages/{image_idx}.jpg'
    infer_detect_picture(model_sparse_rcnn,image_path,f'infer_result/sparse_rcnn/{image_idx}.jpg')
    infer_detect_picture(model_mask_rcnn,image_path,f'infer_result/mask_rcnn/{image_idx}.jpg')

# 来源于互联网的其他图片
folder_path = "data/from_web"
image_file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for image_file in image_file_names:
    infer_detect_picture(model_sparse_rcnn,f'data/from_web/{image_file}',f'infer_result/from_web/sparse_rcnn/{image_file}')
    infer_detect_picture(model_mask_rcnn,f'data/from_web/{image_file}',f'infer_result/from_web/mask_rcnn/{image_file}')

