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
valid_json_path = "./demo_dataset/validation.json"
test_json_path = "./demo_dataset/test.json"
mean_std_path = "./mean_std.json"

batch_size = 32
save_model_path = "./weights/model.pth"

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

model = ResNet18(12,True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_dataset = generate_dataset(train_json_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
valid_dataset = generate_dataset(valid_json_path)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
test_dataset = generate_dataset(test_json_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

def train(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data["image"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Epoch: {}/{}, Step: {}/{}, Loss: {}".format(epoch+1, epochs, i, len(train_loader), loss.item()))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data["image"].to(device), data["label"].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the validation images: {}%".format(100 * correct / total))

train(model, train_loader, valid_loader, criterion, optimizer, epochs=10)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the test images: {}%".format(100 * correct / total))

test(model, test_loader)

# save model
torch.save(model.state_dict(), save_model_path)
