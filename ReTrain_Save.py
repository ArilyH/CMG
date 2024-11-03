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
from torch.utils.data import TensorDataset, ConcatDataset, Subset, WeightedRandomSampler
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

# TRAIN_DIRS['CMG']="CMGTrain_REAL"
# TEST_DIRS['CMG']="CMGTest_REAL"

# TRAIN_DIRS["CMG"]="CMGTrainAdd"
# TEST_DIRS["CMG"]="CMGTestAdd"
data_dir = "./data/"
seed_dir= "./seed"
add_dir = "./data/"
# if MODE=="CMG":
#     add_dir="/home/ubuntu/hbl/LargeD/ProCMG/IN100/CMG_NCOUT"
if not os.path.isdir(add_dir):
    print("File Not Found")
    exit(0)
data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'add': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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


# mode=["few", "medium", ""]
#You could add all modes here and remove the "BREAK" in row 332 to get all the results.
#Or you could use ReTrainInf.py after ReTraining in all modes.
MODES=["PGD"]
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
    dataloaders['add'] = DataLoader(image_datasets['add'], batch_size=32, shuffle=True, pin_memory=True,num_workers=4)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test','add']}
    
    
    num_classes = len(image_datasets['train'].classes)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

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
    
    train_dataset = image_datasets['train']
    add_dataset = image_datasets['add']
    train_weights = [1.0] * len(train_dataset)
    
    # ReSample to get more knowledge of original data
    add_weights = [len(train_dataset) / len(add_dataset)/10] * len(add_dataset)
    sample_weights = train_weights + add_weights
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True) 
    train_all = ConcatDataset([train_dataset, add_dataset])
    dataloaders['train']=DataLoader(train_all, batch_size=32, pin_memory=True,num_workers=4,sampler=sampler)
    epochs=50
    # exit(0)

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / dataset_sizes["train"]:.4f}')
        losses.append(running_loss / dataset_sizes["train"])
        # exp_lr_scheduler.step()
    os.makedirs('./models',exist_ok=True)
    torch.save(model, f'models/resnet50{MODE}.pth')
    # continue
    # test
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
                if cnt_dict[predicted[i].to("cpu").item()] > 50:
                    if predicted[i] == labels[i]:  # MANY类
                        true_positives_many += 1
                    if predicted[i] != labels[i]:  # MANY类
                        false_negatives_many += 1

    # 计算MANY类的召回率
    # recall_many = true_positives_many / (true_positives_many + false_negatives_many)
    recall_many=0
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
            print(f"{key}: {0}, {cnt_dict[key]}, {test_cnt_dict[key]}")
            zcnt+=1
    for key,value in phase_acc.items():
        if key=='few':
            phase_cnt[key]+=zcnt
            if phase_cnt[key]:
                phase_acc[key]/=phase_cnt[key]
        else:
            if phase_cnt[key]:
                phase_acc[key]/=phase_cnt[key]
    accuracy = 100 * correct / total
    print(f'{MODE}-ORG')
    print(f'Accuracy on test data: {accuracy:.2f}%')
    print(f"few: {phase_acc['few']} \n medium: {phase_acc['medium']}\n  many: {phase_acc['many']}")
    print(f'Recall for MANY class: {recall_many:.2f}\n\n')
    
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
                    if cnt_dict[predicted[i].to("cpu").item()] > 50:
                        if predicted[i] == labels[i]:  # MANY类
                            true_positives_many += 1
                        if predicted[i] != labels[i]:  # MANY类
                            false_negatives_many += 1
                    
        for key,value in correct_dict.items():
            correct_dict[key]/=test_cnt_dict[key]
        # print(test_cnt_dict)
        # print(correct_dict)
        # print(true_positives_many)
        # recall_many = true_positives_many / (true_positives_many + false_negatives_many)
        recall_many=0
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
                print(f"{key}: {0}, {cnt_dict[key]}, {test_cnt_dict[key]}")
                zcnt+=1
        for key,value in phase_acc.items():
            if key=='few':
                phase_cnt[key]+=zcnt
                if phase_cnt[key]:
                    phase_acc[key]/=phase_cnt[key]
            else:
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
    break
print(overall_accuracy_dict)
print(few_accuracy_dict)
print(medium_accuracy_dict)
print(many_accuracy_dict)
print(recall_dict)
    # exit(0)
    