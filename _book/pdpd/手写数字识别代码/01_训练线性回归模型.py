import os
import cv2
import paddle
import numpy as np
import paddle.nn.functional as F

from paddle.io import Dataset
from paddle.io import DataLoader

"""
数据集见：
链接：https://pan.baidu.com/s/1u-j-xUKRjNG8irObdjkYbg 
提取码：1f2z
"""

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = os.listdir(path)
    x = np.zeros((len(image_dir), 28 * 28,), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in enumerate(image_dir):

        # 如果不设置cv2.IMREAD_GRAYSCALE会获取3通道的array
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        x[i, :] = img.reshape(-1)
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


def train(model, train_loader):
    model.train()

    opt = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = data[0].astype('float32')
            labels = data[1].astype('float32')

            # 前向计算过程
            predicts = model(images)

            # loss
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练1000批次的数据,打印当前loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            # 后向传播,更新
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


test_dataset = MyDataset('data/test', transform=transform, mode='train')
# train_dataset = MyDataset('data/train', transform=transform, mode='train')

# print('训练集长度', len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for batch_id, data in enumerate(test_loader()):
    images = data[0].astype('float32')
    labels = data[1].astype('float32')
    print(images.shape)  # 16,28*28
    print(labels.shape)  # 16,1
    break

# model = MNIST()
# train(model, train_loader)
# paddle.save(model.state_dict(), './mnist.pdparams')
