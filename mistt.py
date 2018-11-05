import json
from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip
#叶子分类为61类
# NB_CLASSES = 61

#叶子数据集
class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            # path = 'validation_annotations_20181021'
            path = 'D:\challengeai-database/validation_annotations_20181021.json'
        else:
            # path = preffix
            path = 'D:\challengeai-database/train_annotations_20181021.json'
        # path = f'data/{path}.json'
        self.transform = transform

        #读取json文件
        Data = []
        pth = []
        data = json.load(open(path))
        for i in range(len(data)):
            img = data[i]
            Data.append((img['disease_class']))
            pth.append(os.path.join("D:\challengeai-database\_trainingset_20181023\AgriculturalDisease_trainingset\images/"+(img['image_id'])))
        print('lendata:', len(data))
        SumData = pd.DataFrame({'label_id': Data, 'path': pth})
        print(SumData['label_id'])
        print(SumData['path'])
        self.SumData = SumData

    def __len__(self):
        return self.SumData.shape[0]

    def __getitem__(self, idx):
        row = self.SumData.iloc[idx]
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)
        target = row['label_id'] if 'label_id' in row else -1
        return img, target


normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

#数据预处理，进行数据格式转化为tensor
def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

#加了水平的旋转，归一化
def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

#数据预处理之数据增强
def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])
