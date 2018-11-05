import pandas as pd
import random
import matplotlib.image as mpimg
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip
import os
import config as cfg
import numpy as np

class FurnitureDataset(Dataset):
    def __init__(self, preffix:str, transform=None):
        self.preffix = preffix
        self.class_to_ind = dict(zip(cfg.CLASSES, range(cfg.CLASSES_num)))
        self.prepare_data()
        self.transform = transform

    def prepare_data(self):
        class_info = pd.read_table(os.path.join(cfg.Data_dir, 'label.txt'), delim_whitespace=True)
        val_data = []
        train_data = []
        railway_simple = []
        railway_hard = []
        railway_empty = []
        for index, row in class_info.iterrows():
            img_name = row[0]
            img_class = row[1]
            # print(img_name)
            # print(img_class)
            if img_class == 'railway_simple':
                railway_simple.append([img_name, self.class_to_ind[img_class]])
            elif img_class == 'railway_hard':
                railway_hard.append([img_name, self.class_to_ind[img_class]])
            elif img_class == 'railway_empty':
                railway_empty.append([img_name, self.class_to_ind[img_class]])
            else:
                print('Did not find img_class!')
        # val_data.extend(random.sample(railway_simple, int(len(railway_simple)/5))) # 随机分别从simple/hard/empty抽20%作为验证集
        # val_data.extend(random.sample(railway_hard, int(len(railway_hard)/5)))
        # val_data.extend(random.sample(railway_empty, int(len(railway_empty)/5)))
        val_data.extend(railway_simple[:int(len(railway_simple)/5)])
        val_data.extend(railway_hard[:int(len(railway_hard)/5)])
        val_data.extend(railway_empty[:int(len(railway_empty)/5)])
        train_data.extend(railway_simple[int(len(railway_simple)/5):])
        train_data.extend(railway_hard[int(len(railway_hard)/5):])
        train_data.extend(railway_empty[int(len(railway_empty)/5):])
        if self.preffix == 'train':
            self.data = train_data
            print("Get {} data to train the Model!".format(len(self.data)))
            random.shuffle(self.data)
            # 打乱
        if self.preffix == 'val':
            self.data = val_data
            print("Get {} valid data to test!".format(len(self.data)))
# 注意mpimg读入的图像方式rgb,(h,w,c),而深度学习中要对不同通道应用卷积，一般采取(c,h,w),此处并未有加以转换
# 在transforms.ToTensor进行了转换
    def get_image(self, path):
        image = mpimg.imread(path)
        # image读成一个矩阵
        if len(image.shape) != 3 or image.shape[2] != 3:
        # 判断图片是否为3通道，若len(image.shape)=2指只有长和宽，若有三通道，应为[1024,1024,3]
            image = np.stack((image,) * 3, -1)
        image = Image.fromarray(image)
        # 把图片的rgb矩阵变为图像
        if self.transform:
            # 对图像进行预处理操作，如缩放、裁剪等
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)
    # 获取数据集数量
    def __getitem__(self, idx):
        # 获取图像
        # 如果将标签作为字符串给出，则自动生成idx
        row = self.data[idx]
        target = row[1]
        # 0/1/2(即3个类别对应的数字)
        twocpic_name = row[0]
        # 如'003d8fa0-6bf1-40ed-b54c-ac657f8495c5'
        twocpic_path = os.path.join(cfg.twocpic_train_dir, '0000' + str(twocpic_name) + '.jpg')
        #其中选取训练集还是测试集图片由self.preffix == 'xxx'确定
        # twocpic_train_dir = '.\phy-pneumonia/pmm-classification/picture'
        image = self.get_image(twocpic_path)
        return image, target
        #返回图像和图像对应的分类

    # 所有图像不是一次性存储在内存中，而是根据需要进行读取
normalize_torch = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

normalize_05 = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

def preprocess(normalize, image_size):
        # 定义转换方式，transforms.Compose将多个转换函数组合起来使用，先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # 归一化到(0,1)
            normalize
            # 反归一化
        ])
        # transforms.ToTensor:将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。

def preprocess_hflip(normalize, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            HorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

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
