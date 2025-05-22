# MMDetection: Training Mask R-CNN and Sparse R-CNN on PASCAL VOC

This project outlines the steps to set up an MMDetection environment, download and prepare the PASCAL VOC 2007 dataset, train Mask R-CNN and Sparse R-CNN models, perform inference, and visualize training logs using TensorBoard.

## 1. Environment Setup

We'll use Conda to create an isolated Python environment.

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0rc4"

cd mmdetection

pip install -e .
```

**Note:**
*   Ensure you have PyTorch installed, compatible with your CUDA version if you have a GPU. `mmcv` might install a CPU-only or default CUDA version of PyTorch if not already present.
*   The `mmdetection` directory referred to above is the root of your cloned MMDetection repository.

## 2. Dataset Preparation

### 2.1. Download PASCAL VOC 2007

MMDetection provides a script to download common datasets.

```bash
# Ensure you are in the 'mmdetection' root directory
python tools/misc/download_dataset.py --dataset-name voc2007
```
This will download and extract the VOC2007 dataset to `mmdetection/data/VOCdevkit/VOC2007`.

### 2.2. Convert PASCAL VOC to COCO Format

Mask R-CNN and many other models in MMDetection expect the dataset to be in COCO format.

```bash
# Ensure you are in the 'mmdetection' root directory
python tools/dataset_converters/pascal_voc.py \
    data/VOCdevkit \
    --out-dir data/VOCdevkit/coco_format \
    --out-format coco \
    --dataset voc0712 \
    --image-source jpg \
    --label-source txt
```
This command processes the `VOC2007` data (and `VOC2012` if present and specified) located in `data/VOCdevkit` and saves the COCO-formatted annotations to `data/VOCdevkit/coco_format`. The config files we use are typically set up for `voc0712`, meaning they expect both VOC2007 and VOC2012. If you only have VOC2007, you might need to adjust the dataset configurations in the `.py` files or download VOC2012 as well.

## 3. Model Training

We will train two models: Mask R-CNN and Sparse R-CNN. The commands below assume you have 8 GPUs available. If you have a different number of GPUs, change the `8` accordingly (e.g., `1` for a single GPU, but `dist_train.sh` is for distributed training. For single GPU, use `python tools/train.py <CONFIG_FILE>`).

Training logs and checkpoints will be saved to `mmdetection/work_dirs/<CONFIG_NAME>/`.

### 3.1. Train Mask R-CNN

```bash
# Ensure you are in the 'mmdetection' root directory
bash tools/dist_train.sh configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py 8
```

### 3.2. Train Sparse R-CNN

```bash
# Ensure you are in the 'mmdetection' root directory
bash tools/dist_train.sh configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py 8
```

## 4. Inference
```bash
python infer.py
```
**Note:** To modify `infer.py` if you have specific need.

## 5. Log Analysis and TensorBoard Visualization

MMDetection training logs are saved as JSON files. We can convert these to a format TensorBoard can read.
**For Sparse R-CNN:**
```bash
export SPARSE_RCNN_TIMESTAMP=20250514_092228 # Example, set your actual timestamp
python parse_mmdet_log_to_tensorboard.py \
    mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/${SPARSE_RCNN_TIMESTAMP}/vis_data/${SPARSE_RCNN_TIMESTAMP}.json \
    --tb_log_dir mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/${SPARSE_RCNN_TIMESTAMP}/my_tensorboard_logs
```

**For Mask R-CNN:**
```bash
export MASK_RCNN_TIMESTAMP=20250520_110118 # Example, set your actual timestamp
python parse_mmdet_log_to_tensorboard.py \
    mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/${MASK_RCNN_TIMESTAMP}/vis_data/${MASK_RCNN_TIMESTAMP}.json \
    --tb_log_dir mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/${MASK_RCNN_TIMESTAMP}/my_tensorboard_logs
```

### 5.2. Launch TensorBoard

Once the logs are converted, you can visualize them using TensorBoard.

**For Sparse R-CNN:**
```bash
# Replace <SPARSE_RCNN_TIMESTAMP> with the actual timestamp
tensorboard --logdir mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/${SPARSE_RCNN_TIMESTAMP}/my_tensorboard_logs
```

**For Mask R-CNN:**
```bash
# Replace <MASK_RCNN_TIMESTAMP> with the actual timestamp
tensorboard --logdir mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712/${MASK_RCNN_TIMESTAMP}/my_tensorboard_logs
```

Open your web browser and navigate to the URL provided by TensorBoard (usually `http://localhost:6006`).
