import torch
import numpy as np
from torchvision import transforms
import os
src_pth="./DistXplore/DistXplore/dist-guided/GA_100_logits_flws_resnet/100_50"
dirs=os.listdir(src_pth)
dst_pth="./data/Dist"

os.makedirs(dst_pth,exist_ok=True)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

verse_transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])

idx_to_cls={0: 'American alligator, Alligator mississipiensis', 1: 'American coot, marsh hen, mud hen, water hen, Fulica americana', 2: 'Dungeness crab, Cancer magister', 3: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 4: 'agama', 5: 'albatross, mollymawk', 6: 'axolotl, mud puppy, Ambystoma mexicanum', 7: 'bald eagle, American eagle, Haliaeetus leucocephalus', 8: 'banded gecko', 9: 'barn spider, Araneus cavaticus', 10: 'bee eater', 11: 'bittern', 12: 'black and gold garden spider, Argiope aurantia', 13: 'black grouse', 14: 'black swan, Cygnus atratus', 15: 'black widow, Latrodectus mactans', 16: 'boa constrictor, Constrictor constrictor', 17: 'bulbul', 18: 'bustard', 19: 'chambered nautilus, pearly nautilus, nautilus', 20: 'chickadee', 21: 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 22: 'cock', 23: 'common iguana, iguana, Iguana iguana', 24: 'common newt, Triturus vulgaris', 25: 'conch', 26: 'coucal', 27: 'crane', 28: 'crayfish, crawfish, crawdad, crawdaddy', 29: 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 30: 'drake', 31: 'electric ray, crampfish, numbfish, torpedo', 32: 'flamingo', 33: 'flatworm, platyhelminth', 34: 'garden spider, Aranea diademata', 35: 'garter snake, grass snake', 36: 'goldfinch, Carduelis carduelis', 37: 'goldfish, Carassius auratus', 38: 'goose', 39: 'great grey owl, great gray owl, Strix nebulosa', 40: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 41: 'green lizard, Lacerta viridis', 42: 'green mamba', 43: 'green snake, grass snake', 44: 'hammerhead, hammerhead shark', 45: 'harvestman, daddy longlegs, Phalangium opilio', 46: 'hen', 47: 'hermit crab', 48: 'hognose snake, puff adder, sand viper', 49: 'hornbill', 50: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 51: 'hummingbird', 52: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 53: 'jellyfish', 54: 'king snake, kingsnake', 55: 'kite', 56: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 57: 'limpkin, Aramus pictus', 58: 'loggerhead, loggerhead turtle, Caretta caretta', 59: 'lorikeet', 60: 'macaw', 61: 'magpie', 62: 'mud turtle', 63: 'nematode, nematode worm, roundworm', 64: 'night snake, Hypsiglena torquata', 65: 'oystercatcher, oyster catcher', 66: 'peacock', 67: 'pelican', 68: 'prairie chicken, prairie grouse, prairie fowl', 69: 'ptarmigan', 70: 'red-backed sandpiper, dunlin, Erolia alpina', 71: 'redshank, Tringa totanus', 72: 'rock crab, Cancer irroratus', 73: 'scorpion', 74: 'sea anemone, anemone', 75: 'sea lion', 76: 'sea slug, nudibranch', 77: 'sea snake', 78: 'sidewinder, horned rattlesnake, Crotalus cerastes', 79: 'snail', 80: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 81: 'spoonbill', 82: 'spotted salamander, Ambystoma maculatum', 83: 'stingray', 84: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 85: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 86: 'tarantula', 87: 'tench, Tinca tinca', 88: 'terrapin', 89: 'thunder snake, worm snake, Carphophis amoenus', 90: 'tick', 91: 'tiger shark, Galeocerdo cuvieri', 92: 'toucan', 93: 'vine snake', 94: 'wallaby, brush kangaroo', 95: 'water ouzel, dipper', 96: 'whiptail, whiptail lizard', 97: 'white stork, Ciconia ciconia', 98: 'wolf spider, hunting spider', 99: 'wombat'}
for dir in dirs:
    idx=dir.split("_")[1]
    idx=int(idx)
    cls=idx_to_cls[idx]
    cls_dst_pth=os.path.join(dst_pth,cls)
    os.makedirs(cls_dst_pth,exist_ok=True)
    
    cls_dir=os.path.join(src_pth,dir)
    cls_dir=cls_dir+"/best_mmds/"
    if not os.path.isdir(cls_dir):
        continue
    index=0
    items=os.listdir(cls_dir)
    # items=np.load(cls_dir)
    
    for item in items:
        load_pth=os.path.join(cls_dir,item)
        print(load_pth)
        np_item=np.load(load_pth,allow_pickle=True)
        tensor_item=torch.tensor(np_item)
        # print(np_item.shape)
        # exit(0)
        for jpg in tensor_item:
            jpg=jpg.permute(2,0,1)
            jpg=transforms.ToPILImage()(jpg)
            jpg.save(os.path.join(cls_dst_pth,str(index)+".jpg"))
            index+=1