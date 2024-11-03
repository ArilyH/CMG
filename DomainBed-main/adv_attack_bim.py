
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os
import torchvision
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import Counter
import argparse
import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("./")
import domainbed
from domainbed import networks
from domainbed import algorithms

# Define the ResNet architecture for CIFAR-10
class CIFAR_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_ResNet, self).__init__()
        self.resnet = models.resnet20(pretrained=False)
        self.resnet.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Function to preprocess CIFAR-10 data
def cifar_preprocessing(data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(data)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
verse_transform=transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
def empty_processing(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # if len(x.shape)>=4: 
    #     x=np.transpose(x, (0, 3, 1, 2))
    # else:
    #     x=np.transpose(x, (2, 0, 1))
    # 创建逆变换
    verse_transform=transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    x=torch.from_numpy(x)
    x=verse_transform(x)
    return x
# Function to perform adversarial attack
def adv_attack(model, images, labels, method, **kwargs):
    fmodel = PyTorchModel(model, bounds=(0, 1))
    attack = None
    # distance = foolbox.distances.Linfinity
    if method == "fgsm":
        attack = FGSM()
    elif method == "pgd":
        attack = PGD()
    elif method == "cw":
        attack = L2CarliniWagnerAttack()
    
    if attack:
        adversarials = attack(fmodel,images, labels, **kwargs)
        return adversarials
    else:
        return None

def target_adv_attack(tmodel, seeds, labels, method, para_0, para_1):
    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    fmodel = foolbox.PyTorchModel(model=tmodel, bounds=(-55, 55))
    # seeds, xs = foolbox.utils.samples(fmodel, dataset='imagenet', batchsize=len(seeds))
    # distance = foolbox.distances.Linfinity
    # criteria = foolbox.criteria.TargetClass
    # criteria = criteria(target_label)
    # print(seeds)
    seeds=seeds.to('cpu')
    labels=labels.to('cpu')
    if method == "bim":
        attack = foolbox.attacks.L2BasicIterativeAttack()
        adversarials,x,y = attack(fmodel,seeds.to("cuda:0"), labels.to("cuda:0"), epsilons=para_0)
        return adversarials
    elif method == "pgd":
        attack = foolbox.attacks.PGD()
        adversarials,x,y = attack(fmodel,seeds.to("cuda:0"), labels.to("cuda:0"), epsilons=para_0)
        return adversarials
    elif method == "cw":
        attack = foolbox.attacks.carlini_wagner()
        adversarials = attack(fmodel,seeds, labels, learning_rate=para_0, initial_const=para_1)
        return adversarials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="truth and target")
    parser.add_argument("-truth", type=int)
    parser.add_argument("-target", type=int, default=0)
    args = parser.parse_args()

    model=torch.load("../models/ResNet_OH.pth",map_location=torch.device('cpu'))
    # model_dict=model["model_dict"]
    # model = models.resnet50(pretrained=True)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 65)
    # model=model.load_state_dict(model_dict)
    model=model.to("cuda:0")
    model.eval()

    idx_to_cls={0: 'Alarm_Clock', 1: 'Backpack', 2: 'Batteries', 3: 'Bed', 4: 'Bike', 5: 'Bottle', 6: 'Bucket', 7: 'Calculator', 8: 'Calendar', 9: 'Candles', 10: 'Chair', 11: 'Clipboards', 12: 'Computer', 13: 'Couch', 14: 'Curtains', 15: 'Desk_Lamp', 16: 'Drill', 17: 'Eraser', 18: 'Exit_Sign', 19: 'Fan', 20: 'File_Cabinet', 21: 'Flipflops', 22: 'Flowers', 23: 'Folder', 24: 'Fork', 25: 'Glasses', 26: 'Hammer', 27: 'Helmet', 28: 'Kettle', 29: 'Keyboard', 30: 'Knives', 31: 'Lamp_Shade', 32: 'Laptop', 33: 'Marker', 34: 'Monitor', 35: 'Mop', 36: 'Mouse', 37: 'Mug', 38: 'Notebook', 39: 'Oven', 40: 'Pan', 41: 'Paper_Clip', 42: 'Pen', 43: 'Pencil', 44: 'Postit_Notes', 45: 'Printer', 46: 'Push_Pin', 47: 'Radio', 48: 'Refrigerator', 49: 'Ruler', 50: 'Scissors', 51: 'Screwdriver', 52: 'Shelf', 53: 'Sink', 54: 'Sneakers', 55: 'Soda', 56: 'Speaker', 57: 'Spoon', 58: 'TV', 59: 'Table', 60: 'Telephone', 61: 'ToothBrush', 62: 'Toys', 63: 'Trash_Can', 64: 'Webcam'}
    # Load pre-generated adversarial seed data
    adv_seed_dir = "../data/OfficeHome/classwise_data_adv_short/"
    all_class_adv_seeds = []
    for file_index in range(65):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy" % file_index))
        temp_adv_seeds = empty_processing(temp_adv_seeds)
        all_class_adv_seeds.append(temp_adv_seeds)
    # all_class_adv_seeds = np.array(all_class_adv_seeds)

    for idx, class_data in enumerate(all_class_adv_seeds):
        # print(class_data.max().item())
        print(idx)
        print(len(class_data))
        sys.stdout.flush()
        while len(class_data)<50:
            class_data=np.concatenate([class_data,class_data])
        class_data=class_data[:50]
        class_data=torch.tensor(class_data)
        if 1:
            target_list = np.arange(65)
            target_list = np.delete(target_list, idx)
            class_label = torch.ones(len(class_data), dtype=torch.long) * idx
            # print(class_label)
            # print(idx, torch.mean((torch.argmax(model(class_data), dim=1) == class_label).float()))

            index = 0
            for method in ["bim"]:
                adv_save_dir = "../DistXplore/baseline/adv/{}/".format(method)
                adv_npy="adv_data_class_{}_0.npy"
                if(os.path.isfile(os.path.join(adv_save_dir,adv_npy))):
                    continue
                cnt=0
                while(cnt<100):
                    kwargs = {}
                    if method == "bim":
                        kwargs['epsilon'] = 0.05
                        kwargs['iteations'] = 10
                    elif method == "pgd":
                        kwargs['epsilon'] = 0.1
                        kwargs['iterations'] = 10
                    elif method == "cw":
                        kwargs['initial_const'] = 1e-2
                        kwargs['learning_rate'] = 5e-3
                    # exit(0)
                    adv = target_adv_attack(model, class_data, class_label, method, kwargs['epsilon'], 10)
                    cnt+=int(len(adv))
                    if not os.path.exists(adv_save_dir):
                        os.makedirs(adv_save_dir)
                    cls=idx_to_cls[idx]
                    os.makedirs(os.path.join(adv_save_dir,cls),exist_ok=True)
                    
                    end_save_dir=os.path.join(adv_save_dir,cls)
                    if adv is not None:
                        # print(idx, target_label, torch.mean((torch.argmax(model(adv), dim=1) == class_label).float()))
                        # print(Counter(np.argmax(model(adv).numpy(), axis=1)))
                        adv=adv.cpu()
                        for advx in range(adv.shape[0]):
                            img=adv[advx]   
                            img=inverse_transform(img)
                            img = torchvision.transforms.ToPILImage()(img)
                            img.save(os.path.join(end_save_dir,f'imagessAd_{index}.png'))
                            index+=1
                        
                        
                        # np.save(os.path.join(adv_save_dir, "adv_data_class_{}_{}.npy".format(idx, index)), adv.cpu().numpy())
                        # np.save(os.path.join(adv_save_dir, "adv_label_class_{}_{}.npy".format(idx, target_label, index)), class_label.numpy())