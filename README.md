# Cross-View Open World Object Detection in Aerial Imagery

## Installations
````
conda create -n owl python=3.11.4
conda activate owl
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
1. Download the following components of the XView1 dataset from the [official website](https://challenge.xviewdataset.org/download-links): [Training Images] <br>

2. Next, download the **XView1 training labels split into train and validation sets, and modified to COCO format**: 
[train.json](https://drive.google.com/file/d/1-WtULLdnCUL73NuVCheM3cYZfwUB3hem/view?usp=drive_link) 
[val.json](https://drive.google.com/file/d/1IAMYfXmp3L3fzHp-vnN6bRp2BiHxf4ko/view?usp=drive_link).
Move the downloaded .json files to 'datasets/xview/annotations' directory.

## Training
Download baseline weights trained on the LVIS dataset and place in 'models' directory: [Google Drive](https://drive.google.com/file/d/1Bd0K5aOqaNaRdQZNl30kzsbNRq4P_n00/view?usp=drive_link) <br>

Run:
````
python -u train_net.py --num-gpus 2 \
    --config-file configs/owlvit_xview.yaml \
    MODEL.WEIGHTS AerialOpenWorldDetection/models/owlvit_lvis.pth \
    OUTPUT_DIR AerialOpenWorldDetection/output
````

## Evaluation 
Run:
````
python train_net.py --num-gpus 2 \
    --config-file configs/aerialopenworlddetection_xview.yaml \
    --eval-only MODEL.WEIGHTS AerialOpenWorldDetection/output/aerialopenworlddetection_xview.pth
````

## Pre-trained weights
Trained weights on XView1: [Link](https://drive.google.com/file/d/1Nz5KiudBO5PBc3hN1xaBIMF1exMRE2lV/view?usp=sharing)
<br>


## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).


