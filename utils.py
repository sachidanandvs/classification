import torch
import torch.nn as nn
import numpy as np
import json
import cv2

json_files = ["train.json", "validation.json", "test.json"]
dataset_path = "./demo_dataset/"
def calculate_mean_std(dataset_path):
    count = 0
    mean = np.zeros(3)
    square_sum = np.zeros(3)
    
    for json_file in json_files:
        data = json.load(open(dataset_path + json_file))
        for i in range(len(data["images"])):
            img_path = data["images"][i]["file_name"]
            full_path = dataset_path.split("/")
            full_path[-1] = img_path
            full_path = "/".join(full_path)
            img = cv2.imread(full_path).astype(np.float32)/255
            mean += np.mean(img, axis=(0,1))
            square_sum += np.mean(np.square(img), axis=(0,1))
            count += 1
    
    mean = mean/count
    std = np.sqrt(square_sum/count - np.square(mean))


    print("mean: ", mean)
    print("std: ", std)

    result = {"mean":mean.tolist(), "std":std.tolist()}
    with open("mean_std.json", "w") as f:
        json.dump(result, f)


calculate_mean_std(dataset_path)

# another way to calculate mean and std
def cal_mean_std(dataset_path):
    count = 0
    mean = np.zeros(3)
    
    for json_file in json_files:
        data = json.load(open(dataset_path + json_file))
        for i in range(len(data["images"])):
            img_path = data["images"][i]["file_name"]
            full_path = dataset_path.split("/")
            full_path[-1] = img_path
            full_path = "/".join(full_path)
            img = cv2.imread(full_path).astype(np.float32)/255
            mean += np.mean(img, axis=(0,1))
            count += 1
    
    mean = mean/count
    print("mean: ", mean)

    std = np.zeros(3)
    for json_file in json_files:
        data = json.load(open(dataset_path + json_file))
        for i in range(len(data["images"])):
            img_path = data["images"][i]["file_name"]
            full_path = dataset_path.split("/")
            full_path[-1] = img_path
            full_path = "/".join(full_path)
            img = cv2.imread(full_path).astype(np.float32)/255
            std += np.mean(np.square(img-mean), axis=(0,1))
    
    std = np.sqrt(std/count)
    print("std: ", std)

# cal_mean_std(dataset_path)
