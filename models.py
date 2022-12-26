import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes,pretrained=False):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class vgg16(nn.Module):
    def __init__(self, num_classes,pretrained=False):
        super(vgg16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)


class inceptionv3(nn.Module):
    def __init__(self, num_classes,pretrained=False):
        super(inceptionv3, self).__init__()
        self.inceptionv3 = torchvision.models.inception_v3(pretrained=pretrained)
        self.inceptionv3.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.inceptionv3(x)

class efficientnet(nn.Module):
    def __init__(self, num_classes,pretrained=False):
        super(efficientnet, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_v2_m(pretrained=pretrained)
        self.efficientnet.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


