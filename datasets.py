import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import numpy as np
from collections import defaultdict

CROP_SIZE = 500
PER_IMG_LEN = 25

def list_dict():
    return defaultdict(list)

class XViewRandomCrop(Dataset):
    """
    XView random-crop dataset
    """
    def __init__(self, annotations_file, img_dir, mode, transform=None):
        
        self.img_dir = img_dir
        self.mode = mode
        if self.mode == "train":
            crop_transform = A.RandomCrop(width=CROP_SIZE, height=CROP_SIZE)
        else:
            crop_transform = A.CenterCrop(width=CROP_SIZE, height=CROP_SIZE)
            
        self.transform = A.Compose([
            crop_transform,
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
        
        data = json.load(open(annotations_file))
        self.image_id_maps = {im["id"] : im for im in data["images"]}
        category_maps = {cat["id"] : cat["name"] for cat in data["categories"]}
        self.label_to_idx = {label : i+1 for (i, label) in enumerate(set(category_maps.values()))}

        annotations = defaultdict(list_dict)

        for annot in data["annotations"]:
            _id = annot["image_id"]
            x_min, y_min, width, height = annot["bbox"]
            im_width, im_height = self.image_id_maps[_id]["width"], self.image_id_maps[_id]["height"]

            x_center, y_center = x_min + 0.5 * width, y_min + 0.5 * height

            annotations[_id]["bboxes"].append([x_center/im_width, y_center/im_height, width/im_width, height/im_height])
            annotations[_id]["classes"].append(category_maps[annot["category_id"]])
            annotations[_id]["id"].append(annot["id"])
        
        self.annotations = annotations
        self.keys = list(self.annotations.keys())

    def __len__(self):
        if self.mode == "train":
            return len(self.keys) * PER_IMG_LEN
        else:
            return len(self.keys)

    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        annot = self.annotations[self.keys[idx]]
        image_path = os.path.join(self.img_dir, self.image_id_maps[self.keys[idx]]["file_name"])

        bboxes = annot["bboxes"]
        class_labels = annot["classes"]
        annot_id = annot["id"]
        image = Image.open(image_path)


        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        
        eye = np.eye(len(self.label_to_idx)+1)
        one_hot_labels = []

        for label in class_labels:
            one_hot_labels.append(eye[self.label_to_idx[label]])


        return image, list(self.label_to_idx.keys()), one_hot_labels, bboxes, len(self.label_to_idx)+1


def collate_fn(batch):
    inputs = []
    labels = []
    max_size = 1
    for item in batch:
        inputs.append(item[0])
        labels.append(item[1])
        max_size = max(max_size, len(item[2]))
    
    one_hot_labels = []
    bboxes = []
    for item in batch:
        pad_size = max_size - len(item[2])
        one_hot_labels.append(item[2] + pad_size * [np.eye(item[4])[0]])
        bboxes.append(item[3] + pad_size * [(0., 0., 0., 0.)])


    one_hot_labels = np.array(one_hot_labels).astype(np.float32)
    bboxes = np.array(bboxes).astype(np.float32)

    return {'inputs': inputs, 'texts': labels, 'labels': torch.from_numpy(one_hot_labels), 'boxes': torch.from_numpy(bboxes)}

    




if __name__ == "__main__":
    from tqdm import tqdm
    # from transformers import Owlv2Processor
    # from model import XViewDetector

    dataset = XViewRandomCrop(annotations_file="/home/ma293852/Project/Geospatial_atr/datasets/train_updated.json", img_dir="/home/c3-0/datasets/XView/train_images", mode='train')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=2)

    batch = next(iter(dataloader))

    # processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')

    # print(batch['texts'], batch['boxes'].shape)
    # inputs = processor(text=batch['texts'], images=batch['inputs'], return_tensors="pt")
    # model = XViewDetector('google/owlv2-base-patch16-ensemble').cuda()

    # output = model(**{k:v.cuda() for k, v in inputs.items()})

    # print(output['logits'].shape)

    for batch in tqdm(dataloader):
        pass

    # image, labels, one_hot_labels, bboxes = dataset[20]
    # image = Image.fromarray(image)

    # draw = ImageDraw.Draw(image)
    # for bbox, label in zip(bboxes, labels):
    #     xc, yc, w, h = [c*500 for c in bbox]
    #     x0, y0, x1, y1 = [xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h]
    #     draw.rectangle((x0, y0, x1, y1), outline="red")
    #     draw.text((x0, y0), label, font=ImageFont.truetype("OpenSans-Regular.ttf"))

    # image.save("test.jpg")
    # print(one_hot_labels[0])

