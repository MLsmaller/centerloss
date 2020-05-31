#-*- coding:utf-8 -*-
from facenet_pytorch import MTCNN,extract_face
import torch
import shutil
import numpy as np
import os 
import cv2
from PIL import Image,ImageDraw
import time
import sys
from tqdm import tqdm
import argparse
from IPython import embed

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cur_path=os.path.dirname(os.path.realpath(__file__))

def parser():
    parser=argparse.ArgumentParser(description='detect face by MTCNN')
    parser.add_argument('--data_root',type=str,default='vgg2',
                        choices=['lfw','vgg2'],help='the dataset root')
    parser.add_argument('--margin',type=int,default=40,
                        help='the margin pixel around the mtcnn result')
    args=parser.parse_args()
    return args

def detect_face(img_dir='vgg2',margin=40):
    mtcnn=MTCNN(select_largest=True,device=device)
    img_dir=os.path.join(os.path.join(cur_path,img_dir))
    dir_lists=[os.path.join(img_dir,x) for x in os.listdir(img_dir)]
    dir_lists.sort()
    save_path=os.path.join(cur_path,img_dir+'save_path')
    for dir_list in dir_lists[::-1]:
        begin=time.time()
        dir_save=os.path.join(save_path,dir_list.split('/')[-1])
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        else:
            continue
        img_paths=[os.path.join(dir_list,x) for x in os.listdir(dir_list)
                   if x.endswith('.jpg')]
        img_paths.sort()
        print(dir_list)
        
        for img_path in img_paths:
            img=Image.open(img_path)
            start=time.time()
            boxes,_=mtcnn.detect(img)
            draw=ImageDraw.Draw(img)
            w,h=img.size
            if boxes is None:continue
            for box in boxes:
                offset=margin/2
                box[0],box[1],box[2],box[3]= max(box[0]-offset,0),max(box[1]-offset,0),\
                                             min(box[2]+offset,w),min(box[3]+offset,h)
                extract_face(img,box,save_path=os.path.join(dir_save,os.path.split(img_path)[-1]))
                #---output face shape is (160,160,3)
            end=time.time()
            print('img {} has token {:.2f}s'.format(img_path,end-start))

def check_dir(img_dir='vgg2'):
    img_dir=os.path.join(os.path.join(cur_path,img_dir+'save_path'))
    dir_lists=[os.path.join(img_dir,x) for x in os.listdir(img_dir)]
    dir_lists.sort()
    for dir_list in dir_lists:
        img_paths=[os.path.join(dir_list,x) for x in os.listdir(dir_list)
                   if x.endswith('.jpg')]
        img_paths.sort()
        print('the dir {} has {} images'.format(dir_list,len(img_paths)))
        #---删除类别较小的种类
        if len(img_paths)<100:
            shutil.rmtree(dir_list)
#---Image读取图像后的顺序是RGB，而opencv读取图像后顺序(array)是BGR
#----用哪种形式进行显示或者存储就要转换为相应的通道顺序
# numpy_img=np.array(img)[:,:,[2,1,0]]
# cv2.imwrite('./test.jpg',numpy_img)
if __name__=='__main__':
    args=parser()
    # detect_face(img_dir=args.data_root,margin=args.margin)
    check_dir(img_dir=args.data_root)
