import os
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchinfo import summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = {
	'BATCH_SIZE': 4,
    'LEARNING_RATE': 0.001,
    'NUM_EPOCH': 100,
    'IMG_SIZE': 700
        }

### 폴더 카테고리 15개
base_dir = 'D:/abnormal_detection_dataset/mvtec_anomaly_detection/'
# base_dir = 'C:/Users/user/Desktop/test/'


train_path = []
X = np.array([])
Y = np.array([])

def img_read(src, file):
    img = cv2.imread(src+file, cv2.COLOR_BGR2GRAY)
    return img

for path, dir, files in os.walk(base_dir):
    # global img_read
    if 'train' in path:
        if 'good' in path:
            print('path: ', path)

            for file in os.listdir(path):
                # file = '/' + file
                # img_path = cv2.imread(path + '/' + file, cv2.COLOR_BGR2GRAY)
                # img_path = cv2.imread(path + '/' + file, cv2.COLOR_BGR2GRAY)

                # cv2.imshow('1', img_path/255.)
                # cv2.waitKey()
                X = np.append(X, np.array(img_read(path + '/', file)/255.))

# print(X)
print(len(X))
# print(X[0])
# X = np.array(X)
print('X.shape: ', type(X))

X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
print(len(X_train))


    # print(path) # 자기 자신을 포함하여 모든 폴더들의 경로를 다 출력 >>>>>>>>>>>>>>>>>>> 폴더 경로만
    # print(dir) # 자기 자신을 포함하여 폴더 이름만 출력     >>>>>>>>>>>>>>>> 폴더 이름만
    # print(files) # 자기 자신을 포함하여 파일더 이름만 출력 >>>>>>>>>>>>>>> 파일 이름만

X_tensor = torch.Tensor(X)
print(X_tensor.shape)

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
print(X_train.shape)
print(X_test.shape)

class CustomDataset(Dataset):
    def __init__(self, X):
        self.X = X_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        return img

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder # (args['BATCH_SIZE'], 1, args['IMG_SIZE'], args['IMG_SIZE'])
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
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
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)

        return output


X_train = X_train.permute(0, 3, 1, 2)
X_test = X_test.permute(0, 3, 1, 2)

train_dataset = CustomDataset(X_train)
train_loader = DataLoader(train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=0)

test_dataset = CustomDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=0)

model = ConvAutoEncoder()
model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])

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

torch.save(model.state_dict(), './ae_total_data.pth')

net = ConvAutoEncoder()
net.load_state_dict(torch.load('./ae_total_data.pth'))

print(net.parameters)

preds = []
with torch.no_grad():
    val_loss = 0.0
    for val_input in X_test:
        val_output = model(val_input)
        preds.append(val_output)
        v_loss = criterion(val_output, val_input)
        val_loss += v_loss
        # print('preds: ', preds[0].shape) # torch.Size([3, 700, 700])
        # print('X_test: ', X_test[0].shape) # torch.Size([3, 700, 700])
        # print('val_out: ', val_output1.shape) # torch.Size([3, 700, 700])
    print('valdation loss {}'.format(val_loss))

sample_size = 10

test_sample, pred_sample = [], []

with torch.no_grad():
    fig, ax = plt.subplots(2, sample_size, figsize=(15, 4))
    # plt.title('reconstruct')
    for i in range(sample_size):
        # print(X_test.shape)
        val_input1 = X_test[i]

        for val_input1 in X_test:
            val_output1 = model(val_input1)
            # print(val_output1.shape)
            # plt.imshow(val_output1.permute(1,2,0).cpu().numpy())
            # plt.show()

            test_sample.append(val_input1)
            pred_sample.append(val_output1)
            val_input1 = val_input1.permute(1, 2, 0).cpu().numpy()
            # plt.imshow(val_input1)
            # plt.show()
            org_img1 = test_sample[i].permute(1, 2, 0).cpu().numpy()
            rec_img1 = pred_sample[i].permute(1, 2, 0).cpu().numpy()
            # plt.imshow(org_img1)
            # plt.show()

            # print(org_img1.shape)
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(org_img1, cmap=plt.cm.bone)
            ax[1][i].imshow(rec_img1, cmap=plt.cm.bone)
    # plt.show()
    plt.savefig('C:/Users/user/Desktop/origin_data_total_data.png')

test_path = 'D:/abnormal_detection_dataset/mvtec_anomaly_detection/metal_nut/test/custom/'


def img_read(src, file):
    img = cv2.imread(src + file)
    return img


# src 경로에 있는 파일 명을 저장합니다.
files = os.listdir(test_path)

test_X = []
test_Y = []

# 경로와 파일명을 입력으로 넣어 확인하고
# 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다.
for file in files:
    test_X.append(img_read(test_path, file) / 255.)
    test_Y.append(1)  # nomal label : 1

# array로 데이터 변환
test_X = np.array(test_X)
test_Y = np.array(test_Y)

print('Normal shape:', np.shape(test_X))
print(test_X.shape)
print(test_Y.shape)

print(np.shape(test_X))
print(np.shape(test_Y))

test_X = torch.Tensor(test_X)
test_X = test_X.to(device)

# test_X = test_X.unsqueeze(dim=1)
print(test_X.shape)  # torch.Size([2400, 1, 1, 56, 56])

test_Y = torch.Tensor(test_Y)
print(test_Y.shape)

test_X = test_X.permute(0, 3, 1, 2)
print(test_X.shape)

### 13 다른 데이터셋으로 테스트 플랏
sample_size = 10

test_sample_, pred_sample_ = [], []

with torch.no_grad():
    fig, ax = plt.subplots(2, sample_size, figsize=(15, 4))
    # plt.title('reconstruct')
    for i in range(sample_size):
        # print(X_test.shape)
        val_input = test_X[i]

        for val_input in test_X:
            val_output = model(val_input)
            # print(val_output1.shape)
            # plt.imshow(val_output1.permute(1,2,0).cpu().numpy())
            # plt.show()

            test_sample_.append(val_input)
            pred_sample_.append(val_output)
            val_input = val_input.permute(1, 2, 0).cpu().numpy()  # 원래
            val_input = val_input  # .cpu().numpy()
            # plt.imshow(val_input1)
            # plt.show()
            org_img = test_sample_[i].permute(1, 2, 0).cpu().numpy()
            rec_img = pred_sample_[i].permute(1, 2, 0).cpu().numpy()
            # plt.imshow(org_img1)
            # plt.show()

            # print(org_img1.shape)
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(org_img, cmap=plt.cm.bone)
            ax[1][i].imshow(rec_img, cmap=plt.cm.bone)
    # plt.show()
    plt.savefig('C:/Users/user/Desktop/custom_data_total_Data_.png')