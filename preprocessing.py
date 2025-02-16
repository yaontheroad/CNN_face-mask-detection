import os
import numpy as np
import random

import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from torch.utils.data import Dataset, DataLoader
import torch
# from torchvision import transforms

cv2.IMREAD_COLOR
img = cv2.imread('./data/cloth1/2.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Cloth Mask')
plt.show()

img = cv2.imread('./data/ffp2/3.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('FFP2 Mask')
plt.show()

img = cv2.imread('./data/no-mask/1.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('No Mask')
plt.show()

img = cv2.imread('./data/surgical1/129.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Surgical Mask')
plt.show()
plt.close()

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
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.resize(img,img_shape)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data_set.append((img,class_))
		
random.shuffle(data_set) # shuffle data_set
data_X=[]
data_y=[]
for X,y in data_set: # split img and label
    data_X.append(X)
    data_y.append(y)
#     print(y)
data_X = np.array(data_X)
data_y = np.array(data_y) # turn to np.array



print('dataset size is: %d' %len(data_X))


dict_ = {}
for key in data_y:
    dict_[key] = dict_.get(key, 0) + 1
print(dict_)
# dict_ = sorted(dict_.keys())
# dict_.keys()


#plot data statistics
plt.bar(range(len(classes)), list(dict_.values()), tick_label=list(dict_.keys()))
plt.title('Distribution of Classes')
plt.show()

