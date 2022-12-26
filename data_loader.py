import torch
import json
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SyntheticDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.data = json.load(open(dataset_path))
        self.transform = transform

    def __len__(self):
        return len(self.data["images"])
    
    def __getitem__(self,idx):
        img_path = self.data["images"][idx]["file_name"]
        full_path = self.dataset_path.split("/")
        full_path[-1] = img_path
        full_path = "/".join(full_path)
        img = cv2.imread(full_path)
        label = self.data["annotations"][idx]["category_id"]

        if(self.transform):
            img = self.transform(img)

        return {"image":img, "label":label}

