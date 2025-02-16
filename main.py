#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/5 12:07
# @Author  : JY
# @Site    : 
# @File    : main.py
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




############################################################################
num_epochs = 10
num_classes = 4
learning_rate = 0.001

model = CNN()

criterion = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
total_step = len(train_loader)

loss_list = []
#acc_list = []
ave_train_acc = []
test_acc = []
def main():
    for epoch in range(num_epochs):
        acc_list = []
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            images = images.permute(0, 3, 1, 2)
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss_list.append(loss.item())

            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)


            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
        #ave_acc = [].append(sum(acc_list[(epoch*total_step+1):(epoch+1)*total_step])/total_step)
        ave_train_acc.append(sum(acc_list)/len(acc_list))


        #test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.permute(0, 3, 1, 2)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Epoch [{}/{}], Test Accuracy: {} %'
            .format(epoch + 1, num_epochs, (correct / total) * 100))
            test_acc.append(correct / total)

    #plot training loss
    plt.plot(loss_list)
    plt.xlabel('step')
    plt.ylabel('training loss')
    _ = plt.ylim()
    plt.show()

    #plot training accuracy and test accuracy
    plt.plot(ave_train_acc, '-o')
    plt.plot(test_acc, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Accuracy')

    plt.show()

    #save the model
    torch.save(model.state_dict(), 'model.pt')




    # Final Evaluation
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



if __name__ == '__main__':
    main()
