import torch
import numpy as np
import torch
import torch.nn as nn

import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim import lr_scheduler
import numpy as np
import pickle

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建逆变换
verse_transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # print("source {}".format(source))
    # print("target {}".format(target))
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    # print("total.size(0) {}".format(total.size(0)))
    # print("total.size(1) {}".format(total.size(1)))
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # print("L2_distance {}".format(L2_distance))
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    # print("bandwidth {}".format(bandwidth))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # for bandwidth_temp in bandwidth_list:
    #     print("bandwidth_temp {}".format(bandwidth_temp))
    #     print("-L2_distance {}".format(-L2_distance))
    #     print("result {}".format(torch.exp(-L2_distance / bandwidth_temp)))
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # print("kernels val {}".format(kernel_val))
    return sum(kernel_val)


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        # print("kernels {}".format(kernels))
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # print("xx {}".format(XX))
        # print("yy {}".format(YY))
        # print("XY {}".format(XY))
        # print("YX {}".format(YX))
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss

'''
ne_feats: N * sample feats
ae_feats: N * sample feats
'''
# --- Create mmd kernel ---
mmd_loss = MMDLoss()
cls_to_idx={'bluebell': 0, 'buttercup': 1, 'colts_foot': 2, 'cowslip': 3, 'crocus': 4, 'daffodil': 5, 'daisy': 6, 'dandelion': 7, 'fritillary': 8, 'iris': 9, 'lily_valley': 10, 'pansy': 11, 'snowdrop': 12, 'sunflower': 13, 'tigerlily': 14, 'tulip': 15, 'windflower': 16}
idx_to_cls={0: 'bluebell', 1: 'buttercup', 2: 'colts_foot', 3: 'cowslip', 4: 'crocus', 5: 'daffodil', 6: 'daisy', 7: 'dandelion', 8: 'fritillary', 9: 'iris', 10: 'lily_valley', 11: 'pansy', 12: 'snowdrop', 13: 'sunflower', 14: 'tigerlily', 15: 'tulip', 16: 'windflower'}
def extract_features(input, model, device):
    input = input.to(device) 
    model.eval()
    with torch.no_grad():
        x = model.conv1(input)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
    # model.train()
    features = torch.flatten(x, 1)
    features=features.cpu().numpy()
    return features

# --- Calculate MMD ---
def cal_mmd(ae_feats, ne_feats):
    ae_feats=np.array(ae_feats)
    ne_feats=np.array(ne_feats)
    stdl=min(len(ae_feats),len(ne_feats))
    if len(ae_feats)<len(ne_feats):
        s=[]
        idxs=np.random.choice(len(ae_feats),stdl,replace=False)
        for idx in idxs:
            s.append(ae_feats[idx])
        ae_feats=s
    if len(ae_feats)>len(ne_feats):
        s=[]
        idxs=np.random.choice(len(ne_feats),stdl,replace=False)
        for idx in idxs:
            s.append(ne_feats[idx])
        ne_feats=s
    ae_feats = torch.Tensor(np.array(ae_feats))
    ne_feats = torch.Tensor(np.array(ne_feats))
    return mmd_loss(torch.flatten(ae_feats, start_dim=1), torch.flatten(ne_feats, start_dim=1))

model_pth="./models/resnet50.pth"
model = torch.load(model_pth, map_location=torch.device('cpu'))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
data_dir = "./data/train"

os.makedirs(error_data_dir,exist_ok=True)
# test_data_dir=""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# Define your input data
input_data = torch.randn(1, 3, 224, 224)

MODES=["DIST","SENSEI","CMG","PGD","BIM"]
for MODE in MODES:
    print(MODE)
    test_data_dir=f"./data/{MODE}Train/"
    # if MODE=="CMG":
    #     test_data_dir=f"/home/ubuntu/hbl/temp/{MODE}Train_REAL/"
    train_datasets = datasets.ImageFolder(data_dir, data_transforms['train'])
    test_datasets = datasets.ImageFolder(test_data_dir, data_transforms['test'])
    train_dataloaders = DataLoader(train_datasets, batch_size=32,  num_workers=4)
    test_dataloaders=DataLoader(test_datasets, batch_size=32,  num_workers=4)
    cnt_dict=dict()
    for inputs, labels in train_dataloaders:
        for _,l in enumerate(labels):
            cnt_dict[l.item()]=cnt_dict.get(l.item(),0)+1
    errors={}
    for i in range(len(train_datasets.class_to_idx.keys())):
        errors[i]=0
    cnt=0
    with torch.no_grad():
            for inputs, labels in test_dataloaders:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(predicted)):
                    if predicted[i].item()!=labels[i].item():
                        errors[labels[i].item()]+=1
                        cnt+=1
    errorsX=dict()
    for i in range(len(train_datasets.class_to_idx.keys())):
        if cnt_dict[i]<=20:
            # div_tot['few']+=divs[i]
            errorsX['few']=errorsX.get('few',0)+errors[i]
        elif cnt_dict[i]<=50:
            # div_tot['medium']+=divs[i]
            errorsX['medium']=errorsX.get('medium',0)+errors[i]
        else:
            # div_tot['many']+=divs[i]
            errorsX['many']=errorsX.get('many',0)+errors[i]

    E=0
    for i in errorsX.values():
        E+=i
    errorsX={i:j/E for i,j in errorsX.items()}
    print(errors)
    print(errorsX)
    print(E)
            
