import shutil
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
from torch.utils.data import TensorDataset, ConcatDataset, Subset
import random
losses = []
seed = 57  # 选择一个种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果你正在使用多个GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
losses = []
# def onnx_to_h5(output_path ):
#     '''
#     将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
#     '''
#     onnx_model = onnx.load(output_path)
#     tf_rep=prepare(onnx_model)
#     # onnx_model = onnx.load(output_path)
#     # k_model = onnx_to_keras(onnx_model, ['input'])
#     tf.keras.models.save_model(tf_rep, 'models/resnet50_CIFAR10.h5', overwrite=True, include_optimizer=True)    #第二个参数是新的.h5模型的保存地址及文件名
#     # # 下面内容是加载该模型，然后将该模型的结构打印出来
#     # # model = tf.keras.models.load_model('models/kerasModel.h5')
#     # # model.summary()
#     # # print(model)
# MODE=input("Enter mode:")
# print(MODE)
pth={"CMG":"CMG", "DIST":"DIST", "SENSEI":"SENSEI", "BIM":"BIM", "PGD":"PGD"}
TRAIN_DIRS={mode:mode+"Train" for mode in pth}
TEST_DIRS={mode:mode+"Test" for mode in pth}

data_dir = "./data/"
add_dir = "./data/"
if not os.path.isdir(add_dir):
    print("File Not Found")
    exit(0)
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
    ]),
    'add': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建逆变换
verse_transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])


MODES=["CMG","PGD","SENSEI","BIM","DIST"]
device_cnt=torch.cuda.device_count()
overall_accuracy_dict=dict()
few_accuracy_dict=dict()
medium_accuracy_dict=dict()
many_accuracy_dict=dict()
recall_dict=dict()

for MODE in MODES:
    overall_accuracy_dict[MODE]=dict()
    few_accuracy_dict[MODE]=dict()
    medium_accuracy_dict[MODE]=dict()
    many_accuracy_dict[MODE]=dict()
    recall_dict[MODE]=dict()
    
    other_modes=list(set(MODES))
    upper_bound=dict()
    lower_bound=dict()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    image_datasets['add']=datasets.ImageFolder(add_dir+TRAIN_DIRS[MODE],data_transforms['train'])
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, pin_memory=True,num_workers=4) for x in ['train', 'test']}
    # dataloaders['test']=dataloaders['train']
    dataloaders['add'] = DataLoader(image_datasets['add'], batch_size=32, shuffle=True, pin_memory=True,num_workers=4)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test','add']}
    
    
    num_classes = len(image_datasets['train'].classes)
    model = models.resnet50(pretrained=True)
    model_pth=f'/home/ubuntu/hbl/temp/58/models/resnet50{MODE}pth'
    
    model = torch.load(model_pth, map_location=torch.device('cuda:0'))
    
    # 冻结模型的前几层
    # for param in model.parameters():
    #     param.requires_grad = False

    # # 解冻最后的全连接层，使其可以被训练
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    # model_pth="/home/ubuntu/hbl/LargeD/Flower17/models/resnet50_flw17.pth"
    # model = torch.load(model_pth, map_location=torch.device('cuda:0'))
    num_features = model.fc.in_features

    # model.fc = nn.Linear(num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = model.to(device)

    cnt_dict=dict()
    for inputs, labels in dataloaders['train']:
        for _,l in enumerate(labels):
            cnt_dict[l.item()]=cnt_dict.get(l.item(),0)+1

    model.eval()
    correct = 0
    total = 0

    correct_dict=dict()
    test_cnt_dict=dict()
    all_preds = []
    all_labels = []
    
    true_positives_many=0
    false_negatives_many=0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            for i in range(len(predicted)):
                test_cnt_dict[labels[i].to("cpu").item()]=test_cnt_dict.get(labels[i].to("cpu").item(),0)+1
                if predicted[i]==labels[i]:
                    correct_dict[labels[i].to("cpu").item()]=correct_dict.get(labels[i].to("cpu").item(),0)+1
                    correct+=1
                if predicted[i] == labels[i] and cnt_dict[predicted[i].to("cpu").item()] > 50:  # MANY类
                        true_positives_many += 1
                if predicted[i] != labels[i] and cnt_dict[predicted[i].to("cpu").item()] > 50:  # MANY类
                        false_negatives_many += 1

    recall_many = true_positives_many / (true_positives_many + false_negatives_many)
    # recall_many=0
    # continue
    
    # cm = confusion_matrix(all_labels, all_preds)

    for key,value in correct_dict.items():
        correct_dict[key]/=test_cnt_dict[key]
    
    correct_dict=dict(sorted(correct_dict.items(), key=lambda k:k[0], reverse=True))
    phase_cnt={'few':0,"medium":0,"many":0}
    phase_acc={'few':0,"medium":0,"many":0}
    for key,value in correct_dict.items():
        if cnt_dict[key]<=10:
            phase_acc['few']+=value
            phase_cnt['few']+=1
        elif cnt_dict[key]<=50:
            phase_acc['medium']+=value
            phase_cnt['medium']+=1
        else:
            phase_acc['many']+=value
            phase_cnt['many']+=1
        print(f"{key}: {value}, {cnt_dict[key]}")
    zcnt=0
    for key,value in cnt_dict.items():
        if not correct_dict.get(key,0):
            if cnt_dict[key]<=10:
                # phase_acc['few']+=value
                phase_cnt['few']+=1
            elif cnt_dict[key]<=50:
                # phase_acc['medium']+=value
                phase_cnt['medium']+=1
            else:
                # phase_acc['many']+=value
                phase_cnt['many']+=1
            print(f"{key}: {0}, {cnt_dict[key]}, {test_cnt_dict[key]}")
            zcnt+=1
    for key,value in phase_acc.items():
        # if key=='few':
        #     phase_cnt[key]+=zcnt
        #     if phase_cnt[key]:
        #         phase_acc[key]/=phase_cnt[key]
        # else:
        if phase_cnt[key]:
            phase_acc[key]/=phase_cnt[key]
    accuracy = 100 * correct / total
    print(f'{MODE}-ORG')
    print(f'Accuracy on test data: {accuracy:.2f}%')
    print(f"few: {phase_acc['few']} \n medium: {phase_acc['medium']}\n  many: {phase_acc['many']}")
    print(f'Recall for MANY class: {recall_many:.2f}\n\n')
    # break
    overall_accuracy_dict[MODE]['ORG']=accuracy
    few_accuracy_dict[MODE]['ORG']=phase_acc['few']
    medium_accuracy_dict[MODE]['ORG']=phase_acc['medium']
    many_accuracy_dict[MODE]['ORG']=phase_acc['many']
    recall_dict[MODE]['ORG']=recall_many
    
    other_datasets={other :datasets.ImageFolder(add_dir+TEST_DIRS[other],data_transforms['test']) for other in other_modes}
    other_dataloaders={other :DataLoader(other_datasets[other], pin_memory=True, batch_size=32, shuffle=True, num_workers=4) for other in other_modes}
    for other in other_modes:
        print(other)
        true_positives_many=0
        false_negatives_many=0
        model.eval()
        correct = 0
        total = 0

        correct_dict=dict()
        test_cnt_dict=dict()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in other_dataloaders[other]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                for i in range(len(predicted)):
                    test_cnt_dict[labels[i].to("cpu").item()]=test_cnt_dict.get(labels[i].to("cpu").item(),0)+1
                    if predicted[i]==labels[i]:
                        correct_dict[labels[i].to("cpu").item()]=correct_dict.get(labels[i].to("cpu").item(),0)+1
                        correct+=1
                    if predicted[i] == labels[i] and cnt_dict[predicted[i].to("cpu").item()] > 50:  # MANY类
                        true_positives_many += 1
                    if predicted[i] != labels[i] and cnt_dict[predicted[i].to("cpu").item()] > 50:  # MANY类
                        false_negatives_many += 1
                    
        for key,value in correct_dict.items():
            correct_dict[key]/=test_cnt_dict[key]
        # print(test_cnt_dict)
        # print(correct_dict)
        # print(true_positives_many)
        recall_many = true_positives_many / (true_positives_many + false_negatives_many)
        # recall_many=0
        correct_dict=dict(sorted(correct_dict.items(), key=lambda k:k[0], reverse=True))
        phase_cnt={'few':0,"medium":0,"many":0}
        phase_acc={'few':0,"medium":0,"many":0}
        for key,value in correct_dict.items():
            if cnt_dict[key]<=20:
                phase_acc['few']+=value
                phase_cnt['few']+=1
            elif cnt_dict[key]<=50:
                phase_acc['medium']+=value
                phase_cnt['medium']+=1
            else:
                phase_acc['many']+=value
                phase_cnt['many']+=1
            print(f"{key}: {value}, {cnt_dict[key]}")
        zcnt=0
        for key,value in cnt_dict.items():
            if not correct_dict.get(key,0):
                if cnt_dict[key]<=10:
                    # phase_acc['few']+=value
                    phase_cnt['few']+=1
                elif cnt_dict[key]<=50:
                    # phase_acc['medium']+=value
                    phase_cnt['medium']+=1
                else:
                    # phase_acc['many']+=value
                    phase_cnt['many']+=1
                print(f"{key}: {0}, {cnt_dict[key]}, {test_cnt_dict[key]}")
                zcnt+=1
        for key,value in phase_acc.items():
            # if key=='few':
            #     phase_cnt[key]+=zcnt
            #     if phase_cnt[key]:
            #         phase_acc[key]/=phase_cnt[key]
            # else:
            if phase_cnt[key]:
                phase_acc[key]/=phase_cnt[key]
        accuracy = 100 * correct / total
        print(f'{MODE}-{other}')
        print(f'Accuracy on test data: {accuracy:.2f}%')
        print(f"few: {phase_acc['few']} \n medium: {phase_acc['medium']}\n  many: {phase_acc['many']}")
        print(f'Recall for MANY class: {recall_many:.2f}')
        
        overall_accuracy_dict[MODE][other]=accuracy
        few_accuracy_dict[MODE][other]=phase_acc['few']
        medium_accuracy_dict[MODE][other]=phase_acc['medium']
        many_accuracy_dict[MODE][other]=phase_acc['many']
        recall_dict[MODE][other]=recall_many
print(overall_accuracy_dict)
print(few_accuracy_dict)
print(medium_accuracy_dict)
print(many_accuracy_dict)
print(recall_dict)
    # exit(0)
    