import torch
from torch import nn
from models import image_read,Model
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computations")

Tr_data,Tr_dataloader = image_read("emotion recognition/emotion recognition/train")
print("read finished")
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
for epoch in range(200):  
    for inputs, targets in Tr_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.float()
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    loss = torch.Tensor.cpu(loss)
    losses.append(loss)

model.eval()  # 设置模型到评估模式
model_path = 'model.pth'  # 指定保存模型的路径
torch.save(model.state_dict(), model_path)

with torch.no_grad():
    fig, axes = plt.subplots(1,1,figsize = (8,8))

    axes.plot(losses)
    axes.set_title("Loss")

    plt.tight_layout()
    plt.show()