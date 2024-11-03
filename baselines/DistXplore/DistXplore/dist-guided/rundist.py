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
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import TensorDataset,ConcatDataset
import subprocess
import time
import pickle
import shutil
import sys
command = [
    "python3",
    "AttackSetRe.py",
    #Your data path
    "-i", "./classwise_data_adv",
    #Your output path
    "-o", "GA_output/GA_100_logits_Food_resnet/100_50",
    #parameters
    "-pop_num", "100",
    "-subtotal", "50",
    "-type", "mnist",
    # Your model path
    "-model", "./models/resnet50_IN100.pth",
    "-target", "1",
    #parameters
    "-max_iteration", "25"
]
if 1:
    start=time.time()
    file=open("./dist-guided/save_dist_food.txt","w+")
    tfile=open("./dist-guided/timing_dist_food.txt","w+")
    tocnt=0
    alcnt=0
    class_cnt=0
        print("\n\n\n\n")
        print(f"SRC: {label}")
        alcnt=0
        start1=time.time()
        src=label
        tarcnt=0
        
        tarL=np.random.choice(100,100,replace=False)
        for target in tarL:
            with open("./dist-guided/cnt.pickle","wb") as f:
                c=0
                pickle.dump(c,f)
            print(f"TAR: {target}")
            # if tarcnt>=10:
            #     break
            # if not idx[target]:
            #     continue
            out_path="./dist-guided/GA_output/GA_100_logits_Food_resnet/100_50/"
            out_path=os.path.join(out_path,"class_"+str(src)+"_seed_output_"+str(target))
            out_path=os.path.join(out_path,"best_mmds")
            if os.path.isdir(out_path):
                tarcnt+=1
                continue
            if target==src:
                continue
            srcpos=3
            targetpos=15
            srcpath=f'./classwise_data_adv/class_{src}_seed.npy'
            command[srcpos]=srcpath
            command[targetpos]=str(target)
            subprocess.run(command,stdout=file)
            file.flush()
            out_path="./dist-guided/GA_output/GA_100_logits_Food_resnet/100_50/"
            out_path=os.path.join(out_path,"class_"+str(src)+"_seed_output_"+str(target))
            Kout_path=out_path
            out_path=os.path.join(out_path,"best_mmds")
            if not os.path.isdir(out_path):
                shutil.rmtree(Kout_path)
            tocnt=0
            tarcnt+=1
            with open("./dist-guided/cnt.pickle","rb") as f:
                cnt=pickle.load(f)
            alcnt+=cnt
            print("number of generated cases for this src: ",alcnt, file=tfile)
            print(f"timing: {time.time()-start1} cnt:{alcnt}",file=tfile)
            tfile.flush()
            if alcnt>=1300:
                break
        end1=time.time()
        print("timing: ",src," ",end1-start1,file=tfile)
        sys.stdout.flush()
        
    end=time.time()
    print("timing: ",end-start,file=tfile)
    file.close()
    tfile.close()