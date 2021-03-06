from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.heads.det_db_head import DBHead
from ppocr.modeling.necks.db_fpn import DBFPN

from paddle import nn


class BaseModel(nn.Layer):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.backbone = MobileNetV3(in_channels=3, model_name='large', scale=0.5)

        in_channels = self.backbone.out_channels
        self.neck = DBFPN(in_channels=in_channels, out_channels=256)

        self.head = DBHead(in_channels=256, k=50)

    def forward(self, inputs):
        """
        :param inouts: 输入待检测的图片  三通道 像素点必须是32的倍数  [batch size, 3, 320,320]
        :return:
        """
        outputs_01 = self.backbone(inputs)
        outputs_02 = self.neck(outputs_01)
        outputs_03 = self.head(outputs_02)

        # outputs_03中 maps键是最终返回结果   跟原图大小一样
        return outputs_03


if __name__ == '__main__':
    model = BaseModel()
