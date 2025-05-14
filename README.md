# nn_ml_mid_term_task2

配环境
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0rc4"

cd mmdetection
pip install -e .

下载数据
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name voc2012
