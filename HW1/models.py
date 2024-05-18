import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

def image_read(folder_path):

    #将图片规整为48*48
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    #图片读取
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    return dataset, dataloader

class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size=6):
        super(Model, self).__init__()
        # 定义三层隐含层的全连接网络
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        # 通过每个全连接层后应用ReLU激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x