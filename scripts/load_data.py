import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets, utils

import time


tic = time.time()
data_transform = transforms.ToTensor()
dataset = datasets.ImageFolder(root="images",transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
print((time.time() - tic) * 1000000)

def output_to_img(output):
    img = output.cpu().detach().numpy()
    img = img.reshape(img.shape[1:])
    return np.swapaxes(np.swapaxes(np.array(img, dtype=float), 0, 2), 0, 1)

for idx, x in enumerate(dataset_loader):
    tic = time.time()
    input = x[0]
    print((time.time() - tic) * 1000000, end=",")
    tic = time.time()
    utils.save_image(input, "test1.png")
    print((time.time() - tic) * 1000000)

