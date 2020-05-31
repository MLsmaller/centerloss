#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.utils import PairwiseDistance
from IPython import embed

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Deepface(nn.Module):
    def __init__(self,embedding_size,num_classes):
        super(Deepface, self).__init__()
        self.conv1_1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.prelu2_1 = nn.PReLU()
        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.prelu3_1 = nn.PReLU()
        self.conv4_1=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.prelu4_1 = nn.PReLU()
        self.conv5_1=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.prelu5_1 = nn.PReLU()

        self.embedding_size = embedding_size
       
        self.centers=torch.zeros(num_classes,embedding_size).type(torch.FloatTensor).to(device)
        # self.centers=torch.randn(num_classes,embedding_size).type(torch.FloatTensor).to(device)
        #---那些loss计算和针对的特征都是对于倒数第二层的，最后一层是进行分类
        self.fc = nn.Linear(512*6*6, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

    def get_center_loss(self,target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)  #---embedding_size
        #---(batch_size,embedding_size)
        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
        centers_var = self.centers
        
        centers_batch = centers_var.gather(0,target_expand)

        criterion = nn.MSELoss()
        center_loss = criterion(self.features,  centers_batch)
        #->(batch,embedding_size)
        diff = centers_batch - self.features

        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)

        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))

        #---appear_times.view(-1,1)->(batch,1)
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)

        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)

        #∆c_j =(sum_i=1^m δ(yi = j)(c_j − x_i)) / (1 + sum_i=1^m δ(yi = j))
        diff_cpu = alpha * diff_cpu

        for i in range(batch_size):
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss, self.centers

    def l2_norm(self,input):
        #----input->(batch,128(embedding_size))
        input_size = input.size()
        buffer = torch.pow(input, 2)  #--x.pow(2)
        #----normp->(batch,)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        #---x/sqrt(sum(x1+...+xn))  #---对每个image的distance进行归一化
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = F.max_pool2d(x,2)
        x = self.prelu4_1(self.conv4_1(x))
        x1 = F.max_pool2d(x,2)
        x2 = self.prelu5_1(self.conv5_1(x1))
        x=torch.cat((x1,x2),dim=1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #---x.size()->(batch,128)
        self.features=x
        #---对fm进行l2_norm
        # alpha=10
        # self.features_norm = self.l2_norm(x)
        # return self.features_norm*alpha
        return x

    def forward_classifier(self, x):
        features = self.forward(x)
        x = self.classifier(features)
        return x


from torch.nn.parameter import Parameter

class Resnet_Center(nn.Module):
    def __init__(self,embedding_size,num_classes):
        super(Resnet_Center, self).__init__()
        self.model = resnet18()
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512*3*3, 512)
        self.model.fc2 = nn.Linear(512, embedding_size)
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.centers=torch.zeros(num_classes,embedding_size).type(torch.FloatTensor).to(device)
        # self.centers=torch.randn(num_classes,embedding_size).type(torch.FloatTensor).to(device)
        self.num_classes = num_classes

        self.apply(self.weights_init)


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


    def get_center_loss(self,target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)  #---embedding_size

        #---(batch_size,embedding_size)
        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)

        #---一开始设置的centers都是0
        #---self.centers->(num_classes, embedding_size)
        centers_var = Variable(self.centers)
        #---选中gather的维度为0，因此target_expand和centers_var的列维度要相同，
        #---batch_size中记录的是哪一类class，相当于把img对应的centers数据取出来
        #----centers_var->(batch,embedding_size)
        centers_batch = centers_var.gather(0,target_expand)

        criterion = nn.MSELoss()
        #---计算net输出的特征与设定的centers之间的距离，让其之间的距离相近
        #---相当于第一个batch中心的loss就是当前fm的distance，然后第二个batch的时候其fm距离就要与上一个batch中同一个类的fm距离相近
        center_loss = criterion(self.features,  centers_batch)
        
        #->(batch,embedding_size)
        diff = centers_batch - self.features

        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)

        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))

        #---appear_times.view(-1,1)->(batch,1)
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)

        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)

        #∆c_j =(sum_i=1^m δ(yi = j)(c_j − x_i)) / (1 + sum_i=1^m δ(yi = j))
        diff_cpu = alpha * diff_cpu

        for i in range(batch_size):
            #Update the parameters c_j for each j by c^(t+1)_j = c^t_j − α · ∆c^t_j
            #---每个batch中更新每个类的类中心
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss, self.centers

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        #feature for center loss
        x = self.model.fc2(x)
        #---(btach,embedding_size)
        alpha=10
        self.features = x
        self.features_norm = self.l2_norm(x)
        return self.features_norm*alpha

    def forward_classifier(self,x):
        features_norm = self.forward(x)
        x = self.model.classifier(features_norm)
        return x

if __name__ =='__main__':
    #---input 大小为96x96
    x_a=torch.randn(1,3,96,96)
    x_p=torch.randn(1,3,96,96)
    model = Deepface(embedding_size=128,
                      num_classes=20)
    y_a=model(x_a)
    y_p=model(x_p)
    embed()
    l2_dist = PairwiseDistance(2)
    d_p = l2_dist.forward(y_a, y_p)
    print('end')