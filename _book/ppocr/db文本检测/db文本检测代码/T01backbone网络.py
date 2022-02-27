from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.backbones.det_resnet_vd import ResNet

# 也可以从这里引入模型
'https://github.com/PaddlePaddle/PaddleClas/tree/release/2.0/ppcls/modeling/architectures'
import paddle

fake_inputs = paddle.randn([1, 3, 640, 640], dtype="float32")

# 1. 声明Backbone
model_backbone = MobileNetV3()
model_backbone.eval()

# 2. 执行预测
outs = model_backbone(fake_inputs)
print('MobileNetV3模型:')
# 4. 打印输出特征形状
for idx, out in enumerate(outs):
    print("The index is ", idx, "and the shape of output is ", out.shape)

# 1. 声明Backbone
model_backbone = ResNet()
model_backbone.eval()
# 2. 执行预测
outs = model_backbone(fake_inputs)

# 4. 打印输出特征形状
print('restnet模型:')

for idx, out in enumerate(outs):
    print("The index is ", idx, "and the shape of output is ", out.shape)
