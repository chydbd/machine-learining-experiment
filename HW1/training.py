import torch
from torch import nn
from models import image_read,Model
from matplotlib import pyplot as plt
import random

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computations")

Tr_data,Tr_dataloader = image_read("D:/machinelearning datas/emotion recognition/emotion recognition/train")
Ts_data,Ts_dataloader = image_read("D:/machinelearning datas/emotion recognition/emotion recognition/test")
print("read finished")
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
accs = []
for epoch in range(40):  
    count = 0
    acc = 0
    num_equal = 0
    for inputs, targets in Tr_dataloader:
        model.train()
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.float()
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i,(inputs, targets) in enumerate(Ts_dataloader):
        if i > 20:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.float()
        model.eval()
        outputs = model(inputs)
        type_index = torch.argmax(outputs,dim = 1)
        target_index = torch.argmax(targets,dim = 1)
        count += 200
        equal_elements = torch.eq(type_index, target_index)
        num_equal += torch.sum(equal_elements)
    acc += num_equal / count
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    loss = torch.Tensor.cpu(loss)
    losses.append(loss)
    acc = torch.Tensor.cpu(acc)
    accs.append(acc)

model.eval()  # 设置模型到评估模式
model_path = 'model.pth'  # 指定保存模型的路径
torch.save(model.state_dict(), model_path)

with torch.no_grad():
    fig, axes = plt.subplots(1,1,figsize = (8,8))

    axes.plot(losses)
    axes.plot(accs)
    axes.set_title("Loss")

    plt.tight_layout()
    plt.show()