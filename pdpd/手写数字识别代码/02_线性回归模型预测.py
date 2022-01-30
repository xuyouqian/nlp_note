import paddle
import cv2


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


params_file_path = 'mnist.pdparams'
param_dict = paddle.load(params_file_path)
model = MNIST()
model.load_dict(param_dict)

img = cv2.imread('data/test/0_7.png', cv2.IMREAD_GRAYSCALE)
# 展示图片
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# 图片归一化
img = img.reshape(-1) / 255

img = paddle.to_tensor(img).astype('float32')
model.eval()
res = model(img)

print(res)
print("本次预测的数字是", round(res.numpy()[0]))
