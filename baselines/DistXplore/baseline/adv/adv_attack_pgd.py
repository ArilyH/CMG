
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

    model=torch.load("/home/ubuntu/hbl/temp/58/FEW_DATA/models/resnet50_food101.pth",map_location=torch.device('cpu'))
    model=model.to("cuda:0")
    model.eval()

    idx_to_cls={0: 'American alligator, Alligator mississipiensis', 1: 'American coot, marsh hen, mud hen, water hen, Fulica americana', 2: 'Dungeness crab, Cancer magister', 3: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 4: 'agama', 5: 'albatross, mollymawk', 6: 'axolotl, mud puppy, Ambystoma mexicanum', 7: 'bald eagle, American eagle, Haliaeetus leucocephalus', 8: 'banded gecko', 9: 'barn spider, Araneus cavaticus', 10: 'bee eater', 11: 'bittern', 12: 'black and gold garden spider, Argiope aurantia', 13: 'black grouse', 14: 'black swan, Cygnus atratus', 15: 'black widow, Latrodectus mactans', 16: 'boa constrictor, Constrictor constrictor', 17: 'bulbul', 18: 'bustard', 19: 'chambered nautilus, pearly nautilus, nautilus', 20: 'chickadee', 21: 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 22: 'cock', 23: 'common iguana, iguana, Iguana iguana', 24: 'common newt, Triturus vulgaris', 25: 'conch', 26: 'coucal', 27: 'crane', 28: 'crayfish, crawfish, crawdad, crawdaddy', 29: 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 30: 'drake', 31: 'electric ray, crampfish, numbfish, torpedo', 32: 'flamingo', 33: 'flatworm, platyhelminth', 34: 'garden spider, Aranea diademata', 35: 'garter snake, grass snake', 36: 'goldfinch, Carduelis carduelis', 37: 'goldfish, Carassius auratus', 38: 'goose', 39: 'great grey owl, great gray owl, Strix nebulosa', 40: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 41: 'green lizard, Lacerta viridis', 42: 'green mamba', 43: 'green snake, grass snake', 44: 'hammerhead, hammerhead shark', 45: 'harvestman, daddy longlegs, Phalangium opilio', 46: 'hen', 47: 'hermit crab', 48: 'hognose snake, puff adder, sand viper', 49: 'hornbill', 50: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 51: 'hummingbird', 52: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 53: 'jellyfish', 54: 'king snake, kingsnake', 55: 'kite', 56: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 57: 'limpkin, Aramus pictus', 58: 'loggerhead, loggerhead turtle, Caretta caretta', 59: 'lorikeet', 60: 'macaw', 61: 'magpie', 62: 'mud turtle', 63: 'nematode, nematode worm, roundworm', 64: 'night snake, Hypsiglena torquata', 65: 'oystercatcher, oyster catcher', 66: 'peacock', 67: 'pelican', 68: 'prairie chicken, prairie grouse, prairie fowl', 69: 'ptarmigan', 70: 'red-backed sandpiper, dunlin, Erolia alpina', 71: 'redshank, Tringa totanus', 72: 'rock crab, Cancer irroratus', 73: 'scorpion', 74: 'sea anemone, anemone', 75: 'sea lion', 76: 'sea slug, nudibranch', 77: 'sea snake', 78: 'sidewinder, horned rattlesnake, Crotalus cerastes', 79: 'snail', 80: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 81: 'spoonbill', 82: 'spotted salamander, Ambystoma maculatum', 83: 'stingray', 84: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 85: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 86: 'tarantula', 87: 'tench, Tinca tinca', 88: 'terrapin', 89: 'thunder snake, worm snake, Carphophis amoenus', 90: 'tick', 91: 'tiger shark, Galeocerdo cuvieri', 92: 'toucan', 93: 'vine snake', 94: 'wallaby, brush kangaroo', 95: 'water ouzel, dipper', 96: 'whiptail, whiptail lizard', 97: 'white stork, Ciconia ciconia', 98: 'wolf spider, hunting spider', 99: 'wombat'}
    idx_to_cls={0: 'apple_pie', 1: 'baby_back_ribs', 2: 'baklava', 3: 'beef_carpaccio', 4: 'beef_tartare', 5: 'beet_salad', 6: 'beignets', 7: 'bibimbap', 8: 'bread_pudding', 9: 'breakfast_burrito', 10: 'bruschetta', 11: 'caesar_salad', 12: 'cannoli', 13: 'caprese_salad', 14: 'carrot_cake', 15: 'ceviche', 16: 'cheese_plate', 17: 'cheesecake', 18: 'chicken_curry', 19: 'chicken_quesadilla', 20: 'chicken_wings', 21: 'chocolate_cake', 22: 'chocolate_mousse', 23: 'churros', 24: 'clam_chowder', 25: 'club_sandwich', 26: 'crab_cakes', 27: 'creme_brulee', 28: 'croque_madame', 29: 'cup_cakes', 30: 'deviled_eggs', 31: 'donuts', 32: 'dumplings', 33: 'edamame', 34: 'eggs_benedict', 35: 'escargots', 36: 'falafel', 37: 'filet_mignon', 38: 'fish_and_chips', 39: 'foie_gras', 40: 'french_fries', 41: 'french_onion_soup', 42: 'french_toast', 43: 'fried_calamari', 44: 'fried_rice', 45: 'frozen_yogurt', 46: 'garlic_bread', 47: 'gnocchi', 48: 'greek_salad', 49: 'grilled_cheese_sandwich', 50: 'grilled_salmon', 51: 'guacamole', 52: 'gyoza', 53: 'hamburger', 54: 'hot_and_sour_soup', 55: 'hot_dog', 56: 'huevos_rancheros', 57: 'hummus', 58: 'ice_cream', 59: 'lasagna', 60: 'lobster_bisque', 61: 'lobster_roll_sandwich', 62: 'macaroni_and_cheese', 63: 'macarons', 64: 'miso_soup', 65: 'mussels', 66: 'nachos', 67: 'omelette', 68: 'onion_rings', 69: 'oysters', 70: 'pad_thai', 71: 'paella', 72: 'pancakes', 73: 'panna_cotta', 74: 'peking_duck', 75: 'pho', 76: 'pizza', 77: 'pork_chop', 78: 'poutine', 79: 'prime_rib', 80: 'pulled_pork_sandwich', 81: 'ramen', 82: 'ravioli', 83: 'red_velvet_cake', 84: 'risotto', 85: 'samosa', 86: 'sashimi', 87: 'scallops', 88: 'seaweed_salad', 89: 'shrimp_and_grits', 90: 'spaghetti_bolognese', 91: 'spaghetti_carbonara', 92: 'spring_rolls', 93: 'steak', 94: 'strawberry_shortcake', 95: 'sushi', 96: 'tacos', 97: 'takoyaki', 98: 'tiramisu', 99: 'tuna_tartare', 100: 'waffles'}

    # Load pre-generated adversarial seed data
    adv_seed_dir = "/home/ubuntu/hbl/temp/58/FEW_DATA/classwise_data_adv"
    all_class_adv_seeds = []
    for file_index in range(101):
        temp_adv_seeds = np.load(os.path.join(adv_seed_dir, "class_%s_seed.npy" % file_index))
        temp_adv_seeds = empty_processing(temp_adv_seeds)
        all_class_adv_seeds.append(temp_adv_seeds)
    # all_class_adv_seeds = np.array(all_class_adv_seeds)

    for idx, class_data in enumerate(all_class_adv_seeds):
        # print(class_data.max().item())
        if not idx == 100:
            continue
        print(idx)
        print(len(class_data))
        sys.stdout.flush()
        while len(class_data)<50:
            class_data=np.concatenate([class_data,class_data])
        class_data=class_data[:50]
        class_data=torch.tensor(class_data)
        if 1:
            target_list = np.arange(101)
            target_list = np.delete(target_list, idx)
            class_label = torch.ones(len(class_data), dtype=torch.long) * idx
            # print(class_label)
            # print(idx, torch.mean((torch.argmax(model(class_data), dim=1) == class_label).float()))

            index = 0
            for method in ["pgd"]:
                adv_save_dir = "/home/ubuntu/hbl/LargeD/ProCMG/IN100/DistXplore/baseline/{}/".format(method)
                adv_npy="adv_data_class_{}_0.npy"
                if(os.path.isfile(os.path.join(adv_save_dir,adv_npy))):
                    continue
                cnt=0
                while(cnt<1500):
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
                            img.save(os.path.join(end_save_dir,f'imagesAd_{index}.png'))
                            index+=1
                        
                        
                        # np.save(os.path.join(adv_save_dir, "adv_data_class_{}_{}.npy".format(idx, index)), adv.cpu().numpy())
                        # np.save(os.path.join(adv_save_dir, "adv_label_class_{}_{}.npy".format(idx, target_label, index)), class_label.numpy())