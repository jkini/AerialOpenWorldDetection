# Cross-View Open World Object Detection in Aerial Imagery

## Installations
````
conda create -n aerialopenworlddet python=3.11.4
conda activate aerialopenworlddet
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda numpy    
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
````
````
pip install opencv-python
pip install -q git+https://github.com/huggingface/transformers.git
pip install Pillow
pip install pycocotools
pip install tabulate
pip install scikit-image
pip install pandas
````
## Dataset preparation
1. Download the following components of the XView dataset from the [official website](https://challenge.xviewdataset.org/download-links): [Training Images] <br>

2. Next, download the **XView training labels split into train and validation sets, and modified to COCO format**: 
[train.json](https://drive.google.com/drive/folders/1cFSpglnlxWvTxD9_5d3DrR8LldLXqpf5?usp=sharing) 
[val.json](https://drive.google.com/drive/folders/1cFSpglnlxWvTxD9_5d3DrR8LldLXqpf5?usp=sharing).
Move the downloaded .json files to './datasets/' directory.

## Training
````
python -m torch.distributed.run \
    --nproc_per_node=1 train.py \
    --train_annotations_file "./datasets/train.json" \
    --val_annotations_file "./datasets/val.json" \
    --img_dir "./datasets/XView/train_images" \
    --batch_size 8 --num_workers 4 --epochs 100
````

## Evaluation 
````
python -m torch.distributed.run \
    --nproc_per_node=1 eval.py \
    --eval-only MODEL.WEIGHTS ./output/best_epoch.pth
````

## Pre-trained weights
Trained weights on XView: [Link](https://drive.google.com/drive/folders/1cFSpglnlxWvTxD9_5d3DrR8LldLXqpf5?usp=sharing)
<br>

## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).
