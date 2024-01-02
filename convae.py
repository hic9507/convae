# import random
# import pandas as pd
# import numpy as np
# import os
# import cv2
# import torch
#
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
#
# import torchvision.models as models
#
# from tqdm.auto import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score
#
# args = {
# 	'BATCH_SIZE': 5,
#         'LEARNING_RATE': 0.001,
#         'NUM_EPOCH': 20,
#     'IMG_SiZE': 56
#         }
#
#
# src = 'D:/abnormal_detection_dataset/data_3000/'
#
#
# # 이미지 읽기
# def img_read(src, file):
#     img = cv2.imread(src+file, cv2.COLOR_BGR2GRAY)
#     return img
#
# # src 경로에 있는 파일 명을 저장합니다.
# files = os.listdir(src)
#
# X = []
# Y = []
#
# # 경로와 파일명을 입력으로 넣어 확인하고
# # 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다.
# for file in files:
#     X.append(img_read(src,file)/255.)
#     Y.append(1) # nomal label : 1
#
# # array로 데이터 변환
# X = np.array(X)
# Y = np.array(Y)
#
#
#
#
#
# print('Normal shape:',np.shape(X))
#
# import sklearn
# from sklearn.model_selection import train_test_split
#
# # Train set, Test set으로 나누기
# X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=1,shuffle=True)
#
# # 형태를 3차원에서 2차원으로 변경, 첫 번째 인덱스 : 이미지 수, 두 번쨰 인덱스 : 2차원 이미지를 1차원으로 변경 후의 길이
# X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#
# print(np.shape(X_train))
# print(np.shape(X_test))
#
# class CustomDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         img = self.X[idx]
#         labels = self.Y[idx]
#         return img, labels
#
# class ConvAutoEncoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoEncoder, self).__init__()
#
#         # Encoder
#         self.cnn_layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))
#
#         self.cnn_layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))
#
#         # Decoder
#         self.tran_cnn_layer1 = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
#             nn.ReLU())
#
#         self.tran_cnn_layer2 = nn.Sequential(
#             nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
#             nn.Sigmoid())
#
#     def forward(self, x):
#         output = self.cnn_layer1(x)
#         output = self.cnn_layer2(output)
#         output = self.tran_cnn_layer1(output)
#         output = self.tran_cnn_layer2(output)
#
#         return output
#
# train_dataset = CustomDataset(X_train, Y_train)
# train_loader = DataLoader(train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=0)
#
# test_dataset = CustomDataset(X_test, Y_test)
# test_loader = DataLoader(test_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=0)
#
# model = ConvAutoEncoder()
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])
#
# steps = 0
# total_steps = len(train_loader)
# for epoch in range(args['NUM_EPOCH']):
#     running_loss = 0
#     for i, (X_train, _) in enumerate(train_loader):
#         steps += 1
#
#         outputs = model(X_train)
#         loss = criterion(outputs, X_train)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         running_loss += loss.item() * X_train.shape[0]
#
#         if steps % total_steps == 0:
#             model.eval()
#             print('Epoch: {}/{}'.format(epoch + 1, args['NUM_EPOCH']),
#                   'Training loss: {:.5f}..'.format(running_loss / total_steps))
#
#             steps = 0
#             running_loss = 0
#             model.train()

############################################# main.ipynb ##############################################
# -*- coding: utf-8 -*-
import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchinfo import summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


args = {
	'BATCH_SIZE': 5,
        'LEARNING_RATE': 0.001,
        'NUM_EPOCH': 20,
    'IMG_SIZE': 700
        }


# src = 'D:/abnormal_detection_dataset/data_3000/'
src = 'D:/abnormal_detection_dataset/mvtec_anomaly_detection/metal_nut/train/good/'


# 이미지 읽기
def img_read(src, file):
    img = cv2.imread(src+file, cv2.COLOR_BGR2GRAY)
    return img

# src 경로에 있는 파일 명을 저장합니다.
files = os.listdir(src)

X = []
Y = []

# 경로와 파일명을 입력으로 넣어 확인하고
# 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다.
for file in files:
    X.append(img_read(src,file)/255.)
    Y.append(1) # nomal label : 1

# array로 데이터 변환
X = np.array(X)
Y = np.array(Y)

print('Normal shape:',np.shape(X))
print(X.shape)
print(Y.shape)



print(X.shape)
print(Y.shape)
# Train set, Test set으로 나누기
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=1,shuffle=True)

# 형태를 3차원에서 2차원으로 변경, 첫 번째 인덱스 : 이미지 수, 두 번쨰 인덱스 : 2차원 이미지를 1차원으로 변경 후의 길이
# X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print(np.shape(X_train))
print(np.shape(X_test))
print(X_train.shape)
print(Y_train.shape)
print(X.shape)
print(Y.shape)



X_train = torch.Tensor(X_train)

# X_train = X_train.unsqueeze(dim=1)
print(X_train.shape) #torch.Size([2400, 1, 1, 56, 56])

print(X_train.shape)

X_test = torch.Tensor(X_test)
# X_test = X_test.unsqueeze(dim=1)
print(X_test.shape)

Y_train = torch.Tensor(Y_train)
Y_test = torch.Tensor(Y_test)
# data.to(device), target.to(device)
# X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)



class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X_train
        self.Y = Y_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        labels = self.Y[idx]
        return img, labels

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder # (args['BATCH_SIZE'], 1, args['IMG_SIZE'], args['IMG_SIZE'])
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        # Decoder
        self.tran_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU())

        self.tran_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)

        return output



train_dataset = CustomDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=0)

test_dataset = CustomDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=0)

model = ConvAutoEncoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])


from torchvision import models


summary(model)




steps = 0
total_steps = len(train_loader)
for epoch in range(args['NUM_EPOCH']):
    running_loss = 0
    for i, (X_train, _) in enumerate(train_loader):
        steps += 1

        outputs = model(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * X_train.shape[0]

        if steps % total_steps == 0:
            model.eval()
            print('Epoch: {}/{}'.format(epoch + 1, args['NUM_EPOCH']),
                  'Training loss: {:.5f}..'.format(running_loss / total_steps))

            steps = 0
            running_loss = 0
            model.train()

torch.save(model.state_dict(), './ae1.pth')

net = ConvAutoEncoder()
net.load_state_dict(torch.load('./ae1.pth'))

print(net.parameters)



preds = []
with torch.no_grad():
    val_loss = 0.0
    for val_input in X_test:
        val_output = model(val_input)
        preds.append(val_output)
        v_loss = criterion(val_output, X_test)
        val_loss += v_loss
    print('valdation loss {}'.format(val_loss))

sample_size = 10

test_sample, pred_sample = [], []

with torch.no_grad():
    fig, ax = plt.subplots(2, sample_size, figsize=(15, 4))
    plt.title('reconstruct')
    for i in range(sample_size):

        val_input1 = X_test[i]
        for val_input1 in X_test:
            val_output1 = model(val_input1)
            test_sample.append(val_input1)
            pred_sample.append(val_output1)

            org_img1 = test_sample[i].reshape(700, 700)
            rec_img1 = pred_sample[i].reshape(700, 700)

            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(org_img1, cmap=plt.cm.bone)
            ax[1][i].imshow(rec_img1, cmap=plt.cm.bone)
    # plt.show()
    plt.savefig('C:/Users/user/Desktop/origin_data_.png')


#### 다른 데이터셋으로 테스ㅡㅌ
test_path = 'D:/abnormal_detection_dataset/mvtec_anomaly_detection/metal_nut/test/custom/'

def img_read(src, file):
    img = cv2.imread(src+file, cv2.COLOR_BGR2GRAY)
    return img

# src 경로에 있는 파일 명을 저장합니다.
files = os.listdir(test_path)

test_X = []
test_Y = []

# 경로와 파일명을 입력으로 넣어 확인하고
# 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다.
for file in files:
    test_X.append(img_read(test_path,file)/255.)
    test_Y.append(1) # nomal label : 1

# array로 데이터 변환
test_X = np.array(test_X)
test_Y = np.array(test_Y)

print('Normal shape:',np.shape(test_X))
print(test_X.shape)
print(test_Y.shape)

print(np.shape(test_X))
print(np.shape(test_Y))
print(test_X.shape)
print(test_Y.shape)


test_X = torch.Tensor(test_X)
test_X = test_X.to(device)
# test_X = test_X.unsqueeze(dim=1)
print(test_X.shape) #torch.Size([2400, 1, 1, 56, 56])

print(test_X.shape)


test_Y = torch.Tensor(test_Y)
print(test_Y.shape)


preds = []
with torch.no_grad():
    val_loss = 0.0
    for val_input in test_X:
        val_output = model(val_input)
        preds.append(val_output)
        v_loss = criterion(val_output, test_X)
        val_loss += v_loss
    print('valdation loss {}'.format(val_loss))

sample_size = 10

test_sample1, pred_sample1 = [], []

with torch.no_grad():
    fig, ax = plt.subplots(2, sample_size, figsize=(15, 4))
    for i in range(sample_size):

        val_input1 = test_X[i]
        for val_input1 in test_X:
            val_output1 = model(val_input1)
            test_sample1.append(val_input1)
            pred_sample1.append(val_output1)

            org_img1 = test_sample1[i].reshape(700, 700)
            rec_img1 = pred_sample1[i].reshape(700, 700)

            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(org_img1, cmap=plt.cm.bone)
            ax[1][i].imshow(rec_img1, cmap=plt.cm.bone)
    # plt.show()
    plt.savefig('C:/Users/user/Desktop/custom_data_.png')