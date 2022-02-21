import paddle

from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.backbones.det_resnet_vd import ResNet

# 1. 从PaddleOCR中import DBFPN
from ppocr.modeling.necks.db_fpn import DBFPN

# 2. 获得Backbone网络输出结果
fake_inputs = paddle.randn([1, 3, 640, 640], dtype="float32")
# model_backbone = MobileNetV3()
model_backbone = ResNet()
in_channles = model_backbone.out_channels  # [16, 24, 56, 480] & [256, 512, 1024, 2048]

# 3. 声明FPN网络
model_fpn = DBFPN(in_channels=in_channles, out_channels=256)


# 5. 计算得到FPN结果输出
outs = model_backbone(fake_inputs)
fpn_outs = model_fpn(outs)  # [1, 256, 160, 160]

# 6. 打印FPN输出特征形状
print(f"The shape of fpn outs {fpn_outs.shape}")