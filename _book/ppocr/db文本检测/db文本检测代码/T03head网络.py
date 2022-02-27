# 1. 从PaddleOCR中imort DBHead
from ppocr.modeling.heads.det_db_head import DBHead
from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.backbones.det_resnet_vd import ResNet
from ppocr.modeling.necks.db_fpn import DBFPN
import paddle

# 2. 计算DBFPN网络输出结果
fake_inputs = paddle.randn([1, 3, 32, 32], dtype="float32")
model_backbone = MobileNetV3()
in_channles = model_backbone.out_channels
model_fpn = DBFPN(in_channels=in_channles, out_channels=256)
outs = model_backbone(fake_inputs)
fpn_outs = model_fpn(outs)

# 3. 声明Head网络
model_db_head = DBHead(in_channels=256)

# 5. 计算Head网络的输出
db_head_outs = model_db_head(fpn_outs)
print(f"The shape of fpn outs {fpn_outs.shape}")
print(f"The shape of DB head outs {db_head_outs['maps'].shape}")

