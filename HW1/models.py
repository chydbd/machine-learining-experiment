import torch
from torchvision import datasets, transforms

def image_read(folder_path):
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    def custom_target_transform(target):
        return torch.tensor(label_map.get(target, None))

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform, target_transform=custom_target_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    return dataset, dataloader
