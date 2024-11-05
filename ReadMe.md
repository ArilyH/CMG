# CMGFuzz
The Code-Demo of CMGFuzz, a novel distribution-aware neural networks fuzz testing framework that adopts AIGC models to convert training and test data into abstract text modal data and then generate more diverse test cases based on text data.

## Installation
We have tested CMGFuzz based on Python 3.11 on Ubuntu 20.04, theoretically it should also work on other operating systems. To get the dependencies of CMGFuzz, it is sufficient to run the following command.

```
conda env create -f environment.yaml
```

## A User Manual of CMGFuzz
### Seed Selection
Run 
```
python select_seeds_CMG.py
python select_seeds.py
```
to generate seeds(.npy) for BIM, PGD and DistXplore and generate seeds(.jpg) for CMGFuzz.

### CMGFuzz
CMGFuzz is implemented as three components here.
#### Clusering
Run 
```
python Cluster_CMG.py
```
or
```
python Cluster_CMG_More.py
```
to select more representative seeds for CMGFuzz Generation. Here, Cluster_CMG_Fast.py implements just KMeans. And Cluster_CMG_More.py implements KMeans and DBSCAN.

#### Generation
Download a generative model and enter your generative model path into Generate_CMG.py. Or it will download the stable-diffusion-v2 automatically.
and run it with
```
python Generate_CMG.py -dataset_path ./data/c_seed -std 1500 -domain animal
```
to generate test cases.
The meaning of the options are:

-dataset_path: the path of the clustered-dataset

-std: the number of test cases you want to generate

-domain: the domain of input, like "food" or "flower"

#### Data Cleaning
Enter your data path into clean.py, and run it with
```
python clean.py
```

### baselines
We implemented four baseline for evaluation, two adversarial attack methods, one distribution-unaware testing methods and one distribution-aware testing method.
#### ADV
The file attack.py integrate two adversarial attack methods, PGD and BIM. Enter your data and model directions into attack.py, and run it with
```
python attack.py
```
to generate test cases using PGD and BIM.

#### SENSEI
The file trainSENSEI.py implement the baseline SENSEI. Enter your data and model directions into trainSENSEI.py, and run it with
```
python trainSENSEI.py
```
to generate test cases using SENSEI.

#### DistXplore
We re-implemented DistXplore using PyTorch. We have also implemented a user-friendly startup program for it. Modify the command and path in 
```
./baselines/DistXplore/DistXplore/dist-guided/rundist.py
```
and run it with
```
cd ./baselines/DistXplore/DistXplore/dist-guided
python rundist.py
```
to generate test cases using DistXplore.

Note that the test cases generated by DistXplore is in ".npy" format. You can convert it to .jpg by running
```
python save_dist_as_jpg.py
```

### Evaluation
#### Split the Data
We split the generated data into training and test sets and then evaluated the model performance after retraining it using the generated training data. 
For a fair comparison, you can split the generated data into training and test sets and then evaluate the model performance after retraining it using the generated training data. And analyze the model performance on various test datasets.

Just run it with 
```
python split_dataset.py
```
and enter the mode.
#### Neural Coverage
To execute the coverage evaluation, you can run
```
cd ./NeuraL-Coverage
python Coverage_Main.py
```

#### Count Errors
To execute the error evaluation, you can run
```
python Cal_ERR.py
```

#### Diversity visualization
You can perform a visualization to highlight the differences between the data generated by different methods.
```
python paint.py
```

#### ReTrain
You can retrain the models with generated data by running
```
python ./ReTrain_Save.py
```
For efficiency, this file just retrain and save the model for just a methods. You can modify it follow the tips in this file, or run 
```
python ./ReTrainNewInf.py
```
after running it for all the methods.


### Domain Adaptation
We implement all these methods for domain adaptation based on DomainBed.
Run
```
cd ./DomainBed-main
python3 -m domainbed.scripts.trainUDAAddBS --BS_dir=./DIST_LONG/  --data_dir=./OfficeHome --algorithm ERM --dataset OfficeHomeBS --task domain_adaptation --test_env 3 --step 2700 --holdout_fraction=0.5
```
to start a domain adaptation task with generated data. The meaning of the options are:

--BS_dir: the path of the generated data

--data_dir: the path of the original data

--test_env: the test domain

--holdout_fraction: the proportion of the test data, used to split the original data of test domain.


Run
```
cd ./DomainBed-main
python3 -m domainbed.scripts.trainUDAFew  --data_dir=./OfficeHome --algorithm ERM --dataset OfficeHomeFew --task domain_adaptation --test_env 3 --step 2700 --holdout_fraction=0.5 --few_frac=0.2
```
to start a domain adaptation task with none or some original data.

### Gerneration
Note that if you want to generate test cases using DistXplore or BIM, you need a target DNN.
You can find it in 
```
./DomainBed-main/train_output/model.pkl
```
And you should perform some modification to these methods, a code demo is shown in
```
./DomainBed-main/adv_attack_bim.py
```
