# Cross-View Open World Object Detection in Aerial Imagery

## Installations
````
conda create -n dd python=3.9
conda activate dd
conda install pytorch=1.13.0 torchvision -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
````
````
cd DiffusionDetXView
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
````
````
pip install opencv-python
pip install timm
pip install geopandas
````
## Dataset preparation
1. Download the following components of the XView1 dataset from the [official website](https://challenge.xviewdataset.org/download-links): [Training Images] <br>

2. Here, we split the **training images into train and validation sets**, based on splits provided in datasets/xview/train_split.txt and datasets/xview/val_split.txt. Below code run (#a., #b.) just creates symbolic links under the data source 'datasets' directory, instead of moving the images to this directory. <br>

    a. Training split: <br>
    Update create_symlinks_images.py [`source_dir`, `destination_dir`, `file_list_path`] to reflect the relevant details.<br>
    ````
    python -u create_symlinks_images.py 
    ````
    b. Validation split: <br>
    Update create_symlinks_images.py [`source_dir`, `destination_dir`, `file_list_path`] to reflect the relevant details.<br>
    ````
    python -u create_symlinks_images.py 
    ````

3. Next, download the **XView1 training labels split into train and validation sets, and modified to COCO format**: 
[train.json](https://drive.google.com/file/d/1-WtULLdnCUL73NuVCheM3cYZfwUB3hem/view?usp=drive_link) 
[val.json](https://drive.google.com/file/d/1IAMYfXmp3L3fzHp-vnN6bRp2BiHxf4ko/view?usp=drive_link).
Move the downloaded .json files to 'datasets/xview/annotations' directory.

## Training
Download baseline weights trained on the COCO dataset and place in 'models' directory: [Google Drive](https://drive.google.com/file/d/1Bd0K5aOqaNaRdQZNl30kzsbNRq4P_n00/view?usp=drive_link) <br>

Run:
````
python -u train_net.py --num-gpus 2 \
    --config-file configs/diffdet.xview.res50.yaml \
    MODEL.WEIGHTS DiffusionDetXView/models/diffdet_coco_res50.pth \
    OUTPUT_DIR DiffusionDetXView/output
````

## Evaluation 
Run:
````
python train_net.py --num-gpus 2 \
    --config-file configs/diffdet.xview.res50.yaml \
    --eval-only MODEL.WEIGHTS DiffusionDetXView/output/diffdet_xview_res50.pth
````

## Pre-trained weights
Trained weights on XView1: [ResNet50](https://drive.google.com/file/d/1Nz5KiudBO5PBc3hN1xaBIMF1exMRE2lV/view?usp=sharing) 
[SWIN-B](https://drive.google.com/file/d/1yTKCFWiY6YTUzA_Ep2Wwb5MWV20IHsaM/view?usp=sharing)
<br>


## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).


