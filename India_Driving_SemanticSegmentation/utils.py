from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple


from dataloader import n_class
from torch import sum as tsum


def iou(pred, target):
    intersections = [int(tsum((target==pred) * (target==c))) for c in range(n_class)]
    unions = [int(tsum(target==c)) + int(tsum(pred==c)) - intersections[c] for c in range(n_class)]
    ious = [i/u if u > 0 else float('nan') for i, u in zip(intersections, unions)]
    return ious


def pixel_acc(pred, target):
    n_correct = sum([int(tsum((target==pred) * (target==c))) for c in range(n_class - 1)])
    n_total = target.shape[0] * target.shape[1] * target.shape[2]
    n_total -= int(tsum(target==(n_class-1)))
    return n_correct / n_total

wght = [.8,.9,1.1,1.1,.7,1.1, 1.1, 1,2,1,.8,1,1.1,1,1.4,1,1,1.1,1.5,1,2,3,1,.8,.8,1,.6,.6,2]
tots = np.zeros(27)

train_loader = DataLoader(dataset=train_dataset)

def pix_count(label,target):
    tots = np.zeros(27)

    for x,y,z in enumerate(train_loader):
        tots[z] += 1
    
    return tots


