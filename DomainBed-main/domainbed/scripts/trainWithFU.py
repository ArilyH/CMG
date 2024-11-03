# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances
import seaborn as sns
import numpy as np
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
import torch.nn.functional as F
debug=1


def data_to_dict(image_dataset):
    data_by_label = {}  # 创建一个空字典，用于存储按标签组织的数据

    for i in range(len(image_dataset)):
        image, label = image_dataset[i]  # 获取图像和标签
        label = int(label)  # 确保标签为整数

        if label not in data_by_label:
            data_by_label[label] = {"images": [], "labels": []}

        data_by_label[label]["images"].append(image)
        data_by_label[label]["labels"].append(label)
    return data_by_label

def extract_features(input, model, device):
    input = torch.stack(input).to(device) 
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
    model.train()
    features = torch.flatten(x, 1).squeeze(0)
    features=features.cpu().numpy()
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--debug',type=int, default=0)
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    debug=args.debug

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    #device="cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError
    add_env=[0]
    for env_i, env in enumerate(dataset):
        if env_i in add_env:
            train_dataset=env
            add_dataset=dataset[-1][add_env.index(env_i)]
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            
            num_classes=dataset.num_classes
            model.fc = nn.Linear(num_features, num_classes)

            
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            criterion = nn.CrossEntropyLoss()
            model = model.to(device)

            cnt_dict=dict()
            epochs=30
            train_data_by_label = data_to_dict(train_dataset) 
            drop_idxs=[]
            drop_threshold=dict()
            drop_rank=0.05
            cls_drop_threshold=0.2
            droped=dict()
            for epoch in range(epochs): 
                model.train()
                running_loss = 0.0
                add_running_loss = 0.0
                for x in ['train']:
                    d=train_dataset
                    # 生成一个随机的索引序列
                    perm = torch.randperm(len(d))
                    # 使用这个索引序列来创建一个Subset
                    train_dataset = Subset(d, perm.tolist())
                for x in ['add']:
                    d=add_dataset
                    # 生成一个随机的索引序列
                    perm = torch.randperm(len(d))
                    rang=list(range(len(d)))
                    pos_dict=dict(zip(perm.tolist(),rang))
                    # 使用这个索引序列来创建一个Subset
                    add_dataset = Subset(d, perm.tolist())
                    # print(drop_idxs)
                    drop_idxs = [pos_dict[idx] for idx in drop_idxs]
                dataloaders=dict()
                dataloaders['train'] = DataLoader(train_dataset, batch_size=32, num_workers=5)
                dataloaders['add'] = DataLoader(add_dataset, batch_size=32, num_workers=5)
                for inputs, labels in dataloaders['train']:
                    if epoch==0:
                        for _,l in enumerate(labels):
                            drop_threshold[l.item()]=drop_threshold.get(l.item(),0)+1
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                for batch_idx, (inputs, labels) in enumerate(dataloaders['add']):
                    if epoch==0:
                        for _,l in enumerate(labels):
                            drop_threshold[l.item()]=drop_threshold.get(l.item(),0)+1
                    if epoch:
                        batch_size = inputs.size(0)
                        start_idx=batch_idx*batch_size
                        end_idx=(batch_idx+1)*batch_size
                        now_drop_idxs=[i-start_idx for i in drop_idxs if i>=start_idx and i<end_idx]
                        drop_mask = torch.ones(inputs.size(0), dtype=torch.bool)
                        drop_mask[now_drop_idxs] = False
                        org_inputs=inputs
                        org_labels=labels
                        inputs = inputs[drop_mask]
                        labels = labels[drop_mask]
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    add_running_loss += loss.item()
                print(f'Epoch {epoch+1}, Loss: {((running_loss +add_running_loss) / (len(add_dataset)+len(train_dataset)-len(drop_idxs))):.4f}')
                exp_lr_scheduler.step()  
                if epoch==0:
                    for key, value in drop_threshold.items():
                        drop_threshold[key]=value*cls_drop_threshold
                if epoch >=5 and epoch <=epochs:
                    means=dict()
                    sigma=dict()
                    for key,value in train_data_by_label.items():
                        value=value['images']
                        features=[]
                        for v in value:
                            features.append(extract_features([v],model,device))
                        scaler = StandardScaler()
                        scaled_original_features = scaler.fit_transform(features)
                        means[key] = scaler.mean_
                        sigma[key] = np.sqrt(scaler.var_)
                    start=0
                    end=len(add_dataset)
                    to_drop_idxs=[]
                    to_drop_dis=[]
                    start=0
                    end=len(dataset)
                    if end==0:
                        end=len(add_dataset)
                    for i in range(start,end):
                        sample, label = add_dataset[i]
                        if drop_threshold[label]<=0:
                            continue
                        # sample = sample.unsqueeze(0)
                        feature = extract_features([sample], model, device)
                        # scaled_generated_feature = scaler.transform(feature)
                        lower_bound=means[label]-3*sigma[label]
                        upper_bound=means[label]+3*sigma[label]
                        to_drop=((lower_bound > feature) | (feature > upper_bound))
                        if not to_drop.any():
                            continue
                        else:
                            min_dis = np.sum(to_drop)
                            to_drop_idxs.append(i)
                            to_drop_dis.append(min_dis)
                    to_drop_zip=list(zip(to_drop_idxs,to_drop_dis))
                    to_drop_zip=sorted(to_drop_zip, key=lambda x:x[1], reverse=True)
                    to_drop_idxs=[i[0] for i in to_drop_zip if i[0] not in drop_idxs]
                    # print(to_drop_zip)
                    # print(to_drop_idxs)
                    for i in range(int(len(to_drop_idxs)*drop_rank/10)):
                        sample, label=add_dataset[to_drop_idxs[i]]
                        if drop_threshold[label]:
                            drop_idxs.append(to_drop_idxs[i])
                            drop_threshold[label]-=1
                            droped[label]=droped.get(label,0)+1
            print("Clean End")
            selected_idxs=list(range(len(add_dataset)))
            selected_idxs=list(set(selected_idxs).difference(set(drop_idxs)))
            cleaned_add=Subset(add_dataset,selected_idxs)
            train_all=ConcatDataset([train_dataset,cleaned_add])
            dataset[env_i]=train_all
    dataset=dataset[:-1]
    for env_i, env in enumerate(dataset):
        print(env)
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    # for env_i, env in enumerate(dataset):
    #     print(env)
    #     print()
    # exit(0)
    for env_i, env in enumerate(dataset):
        uda = []
        print(env.classes)
        # exit(0)
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        # print((in_[0]))
        # print(out, in_)
        # exit(0)

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    # for i, (env, env_weights) in enumerate(in_splits):
    #     print(len(env))
    #     exit(0)
        # print(type(env))
        # print(env[0][0])
        # print(env.classes)
        # exit(0)

    # print(in_splits[0][0])
    # for i, (env, env_weights) in enumerate(in_splits):
    #     if i in args.test_envs:
    #         continue
    #     print(env[0])
    #     in_splits[0]=(env,in_splits[0][1])
    #     for i, (env, env_weights) in enumerate(in_splits):
    #         if i in args.test_envs:
    #             continue
    #         print(env[0])
    #         exit(0)

    # exit(0)
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    print(0)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ



    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None

    accu=dict()
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc, accu= misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

                accu = sorted(accu.items(), key=lambda d: d[1], reverse=False)
                if(debug):
                    print(accu)
                accu=dict()
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
            

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
