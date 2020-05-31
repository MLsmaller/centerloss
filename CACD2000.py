import torch
import torch.utils.data as data

import numpy as np
import random
import os
from PIL import Image

from IPython import embed

class CACD2000(data.Dataset):

    def __init__(self,data_path,transform=False):
        super(CACD2000,self).__init__
        self.transform=transform
        self.img_dicts,self.fnames=self.get_label(data_path)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,index):
        img_path=self.fnames[index]
        img=Image.open(img_path)
        target=int(self.img_dicts[img_path])
        embed()
        if self.transform:
            img=self.transform(img)
        return img,target

    def get_label(self,data_path):
        img_lists=[os.path.join(data_path,x) for x in os.listdir(data_path)
                   if x.endswith('.jpg')]
        img_names=[os.path.split(x)[-1][3:-9] for x in img_lists]
        img_classes,labels=np.unique(img_names,return_counts=True)

        img_dicts={}
        for img_path in img_lists:
            img_name=os.path.split(img_path)[-1][3:-9]
            label=np.where(img_name==img_classes)[0]
            img_dicts[img_path]=int(label)
        return img_dicts,img_lists

if __name__=='__main__':
    train_data=CACD2000('../CACD2000/')
    train_data.__getitem__(2)
