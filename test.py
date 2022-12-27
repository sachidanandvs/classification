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


test_dataset = generate_dataset(test_json_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

model = ResNet18(12,True).to(device)
model.load_state_dict(torch.load(save_model_path))
model.eval()
criterion = nn.CrossEntropyLoss()


def test(model, test_loader):
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    print("Accuracy: {}%, Loss: {}".format(100 * correct / total, total_loss / total))

test(model, test_loader)


