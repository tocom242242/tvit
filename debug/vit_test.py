import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from vit.vit import ViT
import torch.optim as optim

batch_size = 4

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

from torch.utils.data import random_split

# データセットをランダムに分割
split_ratio = [0.01, 0.99]
train_size = int(len(train_set) * split_ratio[0])
val_size = len(train_set) - train_size
train_dataset, val_dataset = random_split(train_set, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_set = train_set
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViT()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 2
for epoch in range(0, epochs):
    print(epoch)
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_test_loss = 0
    epoch_test_acc = 0
    model.train()
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() / len(train_loader)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        epoch_train_acc += acc / len(train_loader)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item() / len(test_loader)
            test_acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_test_acc += test_acc / len(test_loader)
    print(
        f"Epoch {epoch+1} : train acc. {epoch_train_acc:.2f} train loss {epoch_train_loss:.2f}"
    )
    print(
        f"Epoch {epoch+1} : test acc. {epoch_test_acc:.2f} test loss {epoch_test_loss:.2f}"
    )
