import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data_loader import SyntheticDataset
from models import ResNet18, vgg16
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_json_path = "./demo_dataset/train.json"
test_json_path = "./demo_dataset/test.json"
mean_std_path = "./mean_std.json"

def generate_dataset(data_path,batch_size=32,shuffle=True):

    mean_std = json.load(open(mean_std_path))
    mean = mean_std["mean"]
    std = mean_std["std"]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = SyntheticDataset(data_path, transform)
    return dataset

model = ResNet18(12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_dataset = generate_dataset(train_json_path)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2)
test_dataset = generate_dataset(test_json_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,num_workers=2)

def train(model, train_loader, test_loader,criterion,optimizer,num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data["image"].to(device), data["label"].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    print('Finished Training')

train(model, train_loader, test_loader,criterion,optimizer)
    