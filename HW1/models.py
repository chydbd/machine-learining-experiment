import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from PIL import Image

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        sample, target = super(CustomImageFolder, self).__getitem__(index)
        target_one_hot = torch.nn.functional.one_hot(torch.tensor(target), num_classes=self.num_classes)
        return sample, target_one_hot

def image_read(folder_path):
    # 将图片规整为48*48
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    # 图片读取
    dataset = CustomImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    return dataset, dataloader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3,10,5)
        self.conv2 = nn.Conv2d(10,20,3)
        self.fc1 = nn.Linear(20*20*20,500)
        self.fc2 = nn.Linear(500,6)

    def forward(self, x):
        # 通过每个全连接层后应用ReLU激活函数
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size,-1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
    
def test_image_read(file_path):
    transform = transforms.Compose([
    transforms.Resize((48, 48)),  
    transforms.ToTensor()         
    ])
    image = Image.open(file_path)  # 替换为你的图片路径
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    image_tensor= image_tensor.repeat(1, 3, 1, 1)
    return image_tensor

