import os
import random
from shutil import copyfile
from collections import defaultdict
from torchvision.datasets import ImageFolder
from collections import Counter
import torch
def split_dataset(dataset_dir, train_dir, test_dir, cnt_for_train_dict, cnt_for_test_dict):
    # 创建训练集和测试集目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有类别
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    l=[]
    err_flag=0
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        cnt_for_train=cnt_for_train_dict[cls]
        cnt_for_test=cnt_for_test_dict[cls]
        # 确保每个类别有足够的样本
        # err_flag=0
        # print(len(images))
        if len(images) < cnt_for_train + cnt_for_test:
            l.append(cls)
            err_flag=1
    # if err_flag:
    #     raise ValueError(f"类别 {l} 的样本数量不足")
    # print(err_flag)
    # return 0
    for cls in classes:
        # if not cls=="waffles":
        #     continue
        print(cls)
        cls_dir = os.path.join(dataset_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        cnt_for_train=cnt_for_train_dict[cls]
        # if not cnt_for_train:
        #     cnt_for_train=1
        cnt_for_test=cnt_for_test_dict[cls]
        # 确保每个类别有足够的样本
        err_flag=0
        l=[]
        if not len(images) < cnt_for_train + cnt_for_test:
            train_images = images[:cnt_for_train]
            test_images = images[cnt_for_train:cnt_for_train + cnt_for_test]
        else:
            print(cls,len(images))
            train_images = images[:-cnt_for_test]
            test_images = images[-cnt_for_test:]
        # print(train_images)
        # print(test_images)
        #     l.append(cls)
        #     err_flag=1
        #     raise ValueError(f"类别 {cls} 的样本数量不足")

        # # 分割数据集
        # train_images = images[:cnt_for_train]
        # test_images = images[cnt_for_train:cnt_for_train + cnt_for_test]

        # 复制训练集样本
        train_cls_dir = os.path.join(train_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        for img in train_images:
            copyfile(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))

        # 复制测试集样本
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(test_cls_dir, exist_ok=True)
        for img in test_images:
            copyfile(os.path.join(cls_dir, img), os.path.join(test_cls_dir, img))

org_train='./data/train'
org_test='./data/test'

MODE=input()
pth={"CMG":"CMG", "DIST":"DIST", "SENSEI":"SENSEI", "BIM":"BIM", "PGD":"PGD"}
pth=pth[MODE]

MODE="./data/"+MODE
dataset_dir = MODE
# if MODE=="CMG":
#     dataset_dir = MODE+'Valid_REAL'
train_dir = MODE+'Train'
test_dir = MODE+'Test'

cnt_for_train = {}
cnt_for_test = {}

cnter_train=Counter(ImageFolder(org_train).targets)
cnter_test=Counter(ImageFolder(org_test).targets)
class_to_idx = ImageFolder(org_test).class_to_idx
idx_to_class = {j:i for i,j in class_to_idx.items()}
std=750

cnt_for_train={cls:std for cls in class_to_idx.keys()}
cnt_for_test={cls:300 for cls in class_to_idx.keys()}

split_dataset(dataset_dir, train_dir, test_dir, cnt_for_train, cnt_for_test)
