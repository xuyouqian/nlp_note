# 训练DB文字检测模型

PaddleOCR提供DB文本检测算法，支持MobileNetV3、ResNet50_vd两种骨干网络，可以根据需要选择相应的配置文件，启动训练。

本节以icdar15数据集、MobileNetV3作为骨干网络的DB检测模型（即超轻量模型使用的配置）为例，介绍如何完成PaddleOCR中文字检测模型的训练、评估与测试。

本次实验选取了场景文本检测和识别(Scene Text Detection and Recognition)任务最知名和常用的数据集ICDAR2015。icdar2015数据集的示意图如下图所示：

![img](https://ai-studio-static-online.cdn.bcebos.com/e1b06e0c8e904a2aa412e9eea41f45cce3d58543232948fa88200298fd3cd2e4)


图 icdar2015数据集示意图



数据集格式

```
data/icdar2015/text_localization/train_icdar2015_label.txt
```



```json
icdar_c4_train_imgs/img_61.jpg	[{"transcription": "###", "points": [[427, 293], [469, 293], [468, 315], [425, 314]]}, {"transcription": "###", "points": [[480, 291], [651, 289], [650, 311], [479, 313]]}, {"transcription": "Ave", "points": [[655, 287], [698, 287], [696, 309], [652, 309]]}, {"transcription": "West", "points": [[701, 285], [759, 285], [759, 308], [701, 308]]}, {"transcription": "YOU", "points": [[1044, 531], [1074, 536], [1076, 585], [1046, 579]]}, {"transcription": "CAN", "points": [[1077, 535], [1114, 539], [1117, 595], [1079, 585]]}, {"transcription": "PAY", "points": [[1119, 539], [1160, 543], [1158, 601], [1120, 593]]}, {"transcription": "LESS?", "points": [[1164, 542], [1252, 545], [1253, 624], [1166, 602]]}, {"transcription": "Singapore's", "points": [[1032, 177], [1185, 73], [1191, 143], [1038, 223]]}, {"transcription": "no.1", "points": [[1190, 73], [1270, 19], [1278, 91], [1194, 133]]}]

```

`icdar_c4_train_imgs/img_61.jpg`是图片的路径

后面部分是json格式字符串

`"transcription"`是文本识别后的结果。### 表示图片中的文字看不清不能达标，再目标检测时也可以忽略

`"points"`保存四个坐标定位文本