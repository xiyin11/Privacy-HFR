import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import numpy as np
import torch
import os
import random

def default_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128,128))
    return image

def default_list_reader(label_dict,protocol_path,get_list):
    imgList = []
    get_list(imgList,protocol_path,label_dict)
    random.seed(1)
    random.shuffle(imgList)
    return imgList

class Base_dataset(data.Dataset):
    def __init__(self, root,list_reader=default_list_reader, loader=default_loader ,pid_dict={},istrain=True):
        self.root= root
        self.label_dict=pid_dict
        self.loader    = loader
        self.istrain = istrain
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.list_reader = list_reader
    
    def __getitem__(self, index):
        imgPath, pid, domain= self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        if self.istrain:
            img = self.train_transform(img)
        else:
            img = self.eval_transform(img)
        return img,int(pid),int(domain),imgPath

    def __len__(self):
        return len(self.imgList)