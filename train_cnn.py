import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

data_dir = "data/image"
batch_size = 4

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self,x):
        return self.model(x)

model = CNN()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for imgs,labels in loader:
        opt.zero_grad()
        pred = model(imgs)
        loss = loss_fn(pred, labels)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}/10 Loss:{loss.item():.4f}")

torch.save(model.state_dict(),"models/image/cnn_model.pth")
print("CNN Training Complete!")
