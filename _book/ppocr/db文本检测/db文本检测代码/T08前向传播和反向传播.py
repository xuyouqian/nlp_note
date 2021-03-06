from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.backbones.det_resnet_vd import ResNet
from ppocr.modeling.backbones.det_resnet_vd_sast import ResNet_SAST

from ppocr.modeling.necks.db_fpn import DBFPN
from ppocr.modeling.necks.east_fpn import EASTFPN
from ppocr.modeling.necks.sast_fpn import SASTFPN
from ppocr.modeling.necks.rnn import SequenceEncoder

from ppocr.optimizer import regularizer, optimizer

from ppocr.modeling.heads.det_db_head import DBHead

import paddle
from paddle import nn


class BaseModel(nn.Layer):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.backbone = MobileNetV3(in_channels=3, model_name='large')

        in_channels = self.backbone.out_channels
        self.dbfpn = DBFPN(in_channels=in_channels, out_channels=256)

        self.dbhead = DBHead(in_channels=256)

    def forward(self, inputs):
        """

        :param inouts: 输入待检测的图片  三通道 像素点必须是32的倍数  [batch size, 3, 320,320]
        :return:
        """
        outputs_01 = self.backbone(inputs)
        outputs_02 = self.dbfpn(outputs_01)
        outputs_03 = self.dbhead(outputs_02)

        # outputs_03中 maps键是最终返回结果   跟原图大小一样
        return outputs_03


from T05损失函数 import DBLoss

from paddle.regularizer import L2Decay


def build_optimizer(parameters):
    lr = 0.001
    grad_clip = None
    optim = optimizer.Adam(learning_rate=lr, grad_clip=grad_clip, beta1=0.9, beta2=0.999, weight_decay=L2Decay(0.0001))

    return optim(parameters), lr


if __name__ == "__main__":

    model = BaseModel()
    optim, lr = build_optimizer(model.parameters())
    fake_inputs = paddle.randn([8, 3, 320, 320], dtype='float32')
    labels = [paddle.randn([8, 320, 320])] * 5
    model.train()
    for eopch in range(50):
        for step in range(2):
            outputs = model(fake_inputs)
            loss_fun = DBLoss()
            loss = loss_fun(outputs, labels)
            avg_loss = loss['loss']
            print(avg_loss)
            avg_loss.backward()
            optim.step()

            optim.clear_grad()
