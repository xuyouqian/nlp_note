"""
使用交叉信息熵做损失函数
"""

import os
import cv2
import paddle
import numpy as np
import paddle.nn.functional as F

from paddle.nn import Linear
from paddle.nn import Conv2D
from paddle.io import Dataset
from paddle.nn import MaxPool2D
from paddle.io import DataLoader


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = os.listdir(path)
    x = np.zeros((len(image_dir), 28, 28,), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in enumerate(image_dir):

        # 如果不设置cv2.IMREAD_GRAYSCALE会获取3通道的array
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        x[i, :] = img
        if label:
            y[i] = int(file.split("_")[-1][:-4])  # 取-4作为重点是为了去掉.png 只截取标签
    if label:
        return x, y
    else:
        return x


class MyDataset(Dataset):  # 定义Dataset的子类MyDataset
    def __init__(self, path, transform=None, mode='train'):
        super(MyDataset, self).__init__()

        self.transform = transform
        if mode == "train":
            self.x, self.y = readfile(path, True)
        else:
            self.x = readfile(path, False)
            self.y = None

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def transform(array):
    return array / 255


# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x


# 网络结构部分之后的代码，保持不变
def train(model, train_loader):
    model.train()

    # 使用SGD优化器，learning_rate设置为0.01
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # 训练5轮
    EPOCH_NUM = 10
    # MNIST图像高和宽
    IMG_ROWS, IMG_COLS = 28, 28

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = images.astype('float32').unsqueeze(1)
            labels = labels.astype('int64')

            # 前向计算的过程
            predicts = model(images)

            # 计算损失，取一个批次样本损失的平均值
            loss = paddle.nn.functional.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'cnn_mnist.pdparams')


model = MNIST()
test_dataset = MyDataset('data/test', transform=transform, mode='train')
train_dataset = MyDataset('data/train', transform=transform, mode='train')

print("训练集长度{},,测试集长度{}".format(len(train_dataset), len(test_dataset)))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

train(model, train_loader)
