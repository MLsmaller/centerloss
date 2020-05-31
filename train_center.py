#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import cv2,os
import argparse
import time
import copy
from IPython import embed

from model import Deepface,Resnet_Center
from utils.LFWDataset import LFWDataset
from CACD2000 import CACD2000
from utils.utils import PairwiseDistance,AverageMeter
from utils.utils import adjust_learning_rate,accuracy
from eval_metrics import evaluate,plot_roc,plot_acc

def parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_dir',type=str,default='../CACD2000',
                        help='the train dataset dir')
    parser.add_argument('--val_dir',type=str,default='./lfwsave_path',
                        help='the val dataset dir') 
    parser.add_argument('--val_pairs_dir',type=str,default='./utils/lfw_pairs.txt',
                        help='the val dataset dir')   
    parser.add_argument('--logdir',type=str,default='./log',
                        help='folder to save model logger')
    parser.add_argument('--model',type=str,default='deepface',
                        help='modelname')                        
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')    
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='input batch size for training (default: 128)')                        
    parser.add_argument('--gamma',default=0.1, type=float,help='Gamma update for SGD')                         
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                    help='number of epochs to train (default: 10)')    
    parser.add_argument('--alpha', type=float, default=0.5, help='learning rate of the centers')
    parser.add_argument('--center_loss_weight', type=float, default=0.5, help='weight for center loss')
    parser.add_argument('--embedding_size', type=int, default=128, metavar='ES',
                        help='Dimensionality of the embedding')                        
    args=parser.parse_args()
    return args

args=parser()
if not os.path.exists(args.logdir):os.makedirs(args.logdir)
log_file_path=args.logdir+'/'+time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))+'.log'
# create logger
# logger = Logger(args.logdir)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

train_transform = transforms.Compose([
                         transforms.Resize(96),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         #---归一化到-1-1的范围
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                         ])

val_transform = transforms.Compose([
                         transforms.Resize(96),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                         ])

train_dir = CACD2000(args.train_dir,transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.val_dir,pairs_path=args.val_pairs_dir,
                     transform=val_transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)

print('the number of total train images is {}'.format(len(train_dir)))
data_loaders={'train':train_loader,'val':val_loader}
dataset_sizes = {'train':len(train_loader),'val':len(val_loader)}
l2_dist = PairwiseDistance(2)

if args.model=='deepface':
    model = Deepface(embedding_size=args.embedding_size,num_classes=len(train_dir.classes)).to(device)
else:
    model = Resnet_Center(embedding_size=args.embedding_size,num_classes=len(train_dir.classes)).to(device)
criterion = nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.0)    

epoch_size = len(train_loader) // args.batch_size

print(model)
def train_model():
    step=0
    log_file = open(log_file_path, 'w')
    iteration=0
    lr=args.lr
    top1 = AverageMeter()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss=[]
    val_acc=[]
    for epoch in range(args.epochs):
        # for phase in ['train']:
        for phase in ['train','val']:
            load_t0=time.time()
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
    # top1=AverageMeter()
            if phase=='train':
                for batch_dix,(inputs,labels) in enumerate(data_loaders[phase]):
                    #---(batch,3,96,96)
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase=='train'):
                        prediction=model.forward_classifier(inputs)
                        _,preds=torch.max(F.softmax(prediction),1)
                        center_loss, model.centers = model.get_center_loss(labels, args.alpha)
                        cross_entropy_loss = criterion(prediction,labels)
                        loss = args.center_loss_weight*center_loss + cross_entropy_loss
                        
                        if phase=='train':
                            loss.backward()
                            optimizer.step()
                            iteration+=1
                            # top1.update
                            if epoch in [10,18]:
                                step+=1
                                # lr=adjust_learning_rate(optimizer, args.gamma,step,args.lr
                            # prec=accuracy(F.softmax(prediction), labels, topk=(1,))
                            # top1.update(prec[0], inputs.size(0))
                    load_t1=time.time()
                    running_loss+=loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if iteration%10==0 and phase=='train':
                        log_file.write(
                            'Epoch:' + repr(epoch) + ' || Totel iter ' +repr(iteration) + 
                            ' || L: {:.4f} A: {:.4f} || '.format(loss.item()*inputs.size(0), 
                            torch.sum(preds == labels.data)) +'Batch time: {:.4f}  || '.format(load_t1 - load_t0) + 'LR: {:.8f}'.format(lr) + '\n')
                    train_loss.append(running_loss / dataset_sizes[phase])
                epoch_loss = running_loss / dataset_sizes[phase]
                
                epoch_acc = running_corrects.double() / dataset_sizes[phase]                
                print('epoch {} : {} Loss: {:.4f} Acc: {:.4f}'.format(epoch,phase, epoch_loss, 
                                                        epoch_acc))    
                        
            else:
                targets, distances = [], []
                for batch_dix,(data_a,data_p,labels) in enumerate(data_loaders[phase]):
                    data_a,data_p=data_a.to(device),data_p.to(device)
                    labels=labels.to(device)
                    out_a, out_p = model(data_a), model(data_p)
                    dists = l2_dist.forward(out_a,out_p)
                    distances.append(dists.data.cpu().numpy())
                    targets.append(labels.data.cpu().numpy())
                    # _, _, plot_acc, _,_,_ = evaluate(dists.data.cpu().numpy().tolist(),labels.data.cpu().tolist())
                    
                targets = np.array([sublabel for label in targets for sublabel in label])
                distances = np.array([subdist for dist in distances for subdist in dist])
                tpr, fpr, acc, val, val_std, far = evaluate(distances,targets)
                acc_max=acc.max()
                val_acc.append(acc_max.max())
                # embed()
                print('epoch {} : {}  Acc: {:.4f} '.format(epoch,phase, acc_max))  
                log_file.write('epoch {} : {}  Acc: {:.4f} '.format(epoch,phase, acc_max))
                plot_roc(fpr, tpr, figure_name = './log/roc_valid_epoch_{}.png'.format(epoch))
                
                if acc_max>best_acc:
                    best_acc=acc_max
                    best_model_wts=copy.deepcopy(model.state_dict())

    plot_acc(train_loss,val_acc,'./loss.png')
    torch.save({'epoch':epoch+1,
                'state_dict':model.state_dict(),
                'centers':model.centers,},
                './centers_{}.pth'.format(best_acc))
if __name__=='__main__':
    train_model()