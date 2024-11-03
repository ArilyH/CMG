import torch
import tool
import coverage
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_channel=3
image_size=224


model_pth="./models/resnet50_food101.pth"
model = torch.load(model_pth, map_location=torch.device('cpu'))
model=model.to(device)

# 0. Get layer size in model
input_size = (1, image_channel, image_size, image_size)
random_input = torch.randn(input_size).to(device)
layer_size_dict = tool.get_layer_output_sizes(model, random_input)


criterion = coverage.NBC(model, layer_size_dict)

#Your Training Data
data_dir = "./data/train"
image_datasets = datasets.ImageFolder(data_dir, data_transforms['test'])
train_loader = DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=4)
criterion.build(train_loader)

# MODE=input("PLEASE ENTER MODE:")
txt=[]
for MODE in ["BIM","CMG","SENSEI","PGD","DIST"]:
    pth={"CMG":"CMG", "DIST":"DIST", "SENSEI":"SENSEI", "BIM":"BIM", "PGD":"PGD"}
    MODE=pth[MODE]
    # if MODE=="CMG":
    #     data_dir = f"/home/ubuntu/hbl/temp/{MODE}Train_REAL/"
    # else:
    data_dir = f"./data/{MODE}Train/"
    image_datasets = datasets.ImageFolder(data_dir, data_transforms['train'])
    test_loader = DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=4)
    criterion = coverage.NBC(model, layer_size_dict)
    criterion.build(train_loader)
    criterion.assess(test_loader)
    nbc = criterion.current

    criterion = coverage.SNAC(model, layer_size_dict)
    criterion.build(train_loader)
    criterion.assess(test_loader)
    snac = criterion.current

    criterion = coverage.KMNC(model, layer_size_dict, hyper=100)
    criterion.build(train_loader)
    criterion.assess(test_loader)
    kmnc = criterion.current

    criterion = coverage.TKNC(model, layer_size_dict, hyper=3)
    criterion.build(train_loader)
    criterion.assess(test_loader)
    tknc = criterion.current

    # criterion = coverage.TKNP(model, layer_size_dict, hyper=3)
    # criterion.build(train_loader)
    # criterion.assess(test_loader)
    # tknp = criterion.current
    txt.append(f"{MODE}: nbc:{nbc} snac:{snac} kmnc:{kmnc} tknc:{tknc}")
for i in txt:
    print(i)