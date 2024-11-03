import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans
from torchvision.utils import save_image
import shutil
from PIL import Image

def getAllChildren(path):
    res = []
    isChidren = 1
    for file_path in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_path)):
            res += getAllChildren(os.path.join(path, file_path))
            isChidren = 0
    if isChidren:
        res += [path]
    return res

model = torchvision.models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
])

input_path = './data/seed'
output_path = './data/c_seed'
data_paths = getAllChildren(input_path)
print("Clustered: ", data_paths)

for data_path in data_paths:
    if 'tmp' in data_path:
        continue
    print(data_path)
    output_dir = os.path.join(output_path, data_path.split("/")[-1])
    output_dir = os.path.join(output_dir, 'clusRes')

    tmp_path = './data/tmp'
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    tmp_parents_dir = os.path.join(tmp_path, "parent")
    if not os.path.isdir(tmp_parents_dir):
        os.mkdir(tmp_parents_dir)
    cnt = 0
    for img_path in os.listdir(data_path):
        img_path = os.path.join(data_path, img_path)
        img = Image.open(img_path)
        res_path = os.path.join(tmp_parents_dir, "result" + str(cnt) + ".jpg")
        img.save(res_path)
        cnt += 1

    dataset = torchvision.datasets.ImageFolder(tmp_path, transform=transform)

    features = []
    for image, _ in dataset:
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            feature = model(image)
        features.append(feature.squeeze().numpy())

    features = torch.tensor(features)
    if len(features) >= 3:
        kmeans = KMeans(n_clusters=3)
        cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_

        clusters = [[], [], []]
        for i in range(len(cluster_labels)):
            clusters[cluster_labels[i]].append(i)

        for i in range(len(clusters)):
            if len(clusters[i]) < 4:
                clusters[i] = np.random.choice(clusters[i], 4, replace=True)
            else:
                distances = np.linalg.norm(features[clusters[i]] - cluster_centers[i], axis=1)
                sorted_indices = np.argsort(distances)
                clusters[i] = np.array(clusters[i])[sorted_indices[:4]]
    else:
        clusters = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] if len(features) == 1 else [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]

    print(clusters)
    os.makedirs(output_dir, exist_ok=True)
    for i, (image, _) in enumerate(dataset):
        cluster_label = cluster_labels[i]
        if i in clusters[cluster_label]:
            cluster_dir = os.path.join(output_dir, f'Cluster_{cluster_label}')
        else:
            cluster_dir = os.path.join(output_dir, f'Cluster_{cluster_label}', 'rest')
        os.makedirs(cluster_dir, exist_ok=True)
        image_name = f'image_{i}.jpg'
        image_path = os.path.join(cluster_dir, image_name)
        save_image(inverse_transform(image), image_path)

    shutil.rmtree(tmp_path)
    print(data_path)
