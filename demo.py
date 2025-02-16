#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 15:00
# @Author  : JY
# @Site    : 
# @File    : demo.py
# @Software: PyCharm

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from cv2 import cv2
import random
import numpy as np
from CNN import CNN

# Load data
data = './data/' # data path
img_shape = (128, 128) # pic shape
classes = ['cloth_mask', 'FFP2_mask', 'no-mask', 'surgical_mask' ] # data classes
data_set = [] # data set
for i in enumerate(os.listdir(data)):
    path = os.path.join(data,i[1])
    for j in os.listdir(path):
        # cloth_mask: 0; FFP2_mask: 1; no-mask: 2; surgical_mask:3
        if i[0] in [0,1]:
            class_ = 0
        elif i[0] in [2,3]:
            class_ = 1
        elif i[0] in [4,5]:
            class_ = 2
        else:
            class_ = 3
        img = cv2.imread(os.path.join(path,j),cv2.IMREAD_UNCHANGED)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.resize(img,img_shape)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data_set.append((img,class_))

#Build data set
data_X = []
data_y = []
for X, y in data_set:  # split img and label
    data_X.append(X)
    data_y.append(y)

data_X = np.array(data_X)
data_y = np.array(data_y)  # turn to np.array

# train test split
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

#Package data
# TrainDataset
class TrainDataset(Dataset):
    def __init__(self, isTransform=None):
        self.train_X = train_X
        self.train_y = train_y

        self.isTransform = isTransform

    #  Normalization
    def transform(self, X):
        X = X / 225.
        return X

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, idx):
        X = torch.from_numpy(np.array(self.train_X[idx]))
        if self.isTransform:
            X = self.transform(X)

        y = torch.from_numpy(np.array(self.train_y[idx]))

        return X, y


# TestDataset
class TestDataset(Dataset):
    def __init__(self, isTransform=None):
        self.test_X = test_X
        self.test_y = test_y

        self.isTransform = isTransform

    #  Normalization
    def transform(self, X):
        X = X / 225.
        return X

    def __len__(self):
        return len(self.test_X)

    def __getitem__(self, idx):
        X = torch.from_numpy(np.array(self.test_X[idx]))
        if self.isTransform:
            X = self.transform(X)

        y = torch.from_numpy(np.array(self.test_y[idx]))

        return X, y




train_data_set = TrainDataset(isTransform=True)
train_loader = DataLoader(train_data_set, batch_size=32, shuffle=False)
#images, label = iter(train_loader).next()



test_data_set = TestDataset(isTransform=True)
test_loader = DataLoader(test_data_set, batch_size=len(test_X), shuffle=False)
#t_images, t_label = iter(train_loader).next()


model = CNN()

#load the model
model.load_state_dict(torch.load('model.pt'))

#one example
 #show the picture
sample_img = cv2.imread('./378.png')
plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
plt.show()

# preprocessing the sample image
sample_img = cv2.resize(sample_img, (128,128))
sample_img = sample_img / 225
sample_img = torch.from_numpy(np.array(sample_img))
sample_img = sample_img.permute(2, 0, 1)
sample_img = sample_img.unsqueeze(0).float()
output = model(sample_img)
_, predicted_label = torch.max(output.data, 1)
print('Predicted class is', classes[predicted_label])



# Evaluation
model.eval()
pred_list = []
accu_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.permute(0, 3, 1, 2)
        _, pred = torch.max(model(images).data, 1)
        pred_list.extend(pred.detach().cpu().numpy())
        accu_list.extend(labels.detach().cpu().numpy())
print(classification_report(pred_list, accu_list,
                            target_names=['cloth_mask', 'FFP2_mask', 'no-mask', 'surgical_mask']))

cm = confusion_matrix(pred_list, accu_list, )
ConfusionMatrixDisplay(cm, display_labels=['cloth_mask', 'FFP2_mask', 'no-mask', 'surgical_mask']).plot()
plt.title('Confusion Matrix')
plt.show()