import torch
from torch import nn
from models import image_read,Model

Tr_data,Tr_dataloader = image_read("emotion recognition/emotion recognition/train")
model = Model(input_size = 48*48, hidden_size1 = 16, hidden_size2 = 16, hidden_size3 = 16, output_size=6)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):  # 进行10个训练周期
    for inputs, targets in Tr_dataloader:
        inputs = inputs.view(inputs.size(0), -1)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')