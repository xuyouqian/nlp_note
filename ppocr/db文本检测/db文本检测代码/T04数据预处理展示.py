from paddleocr.ppocr.data.imaug.operators import DecodeImage, NormalizeImage, ToCHWImage, KeepKeys
from paddleocr.ppocr.data.imaug.label_ops import DetLabelEncode

from paddleocr.ppocr.data.imaug.iaa_augment import IaaAugment
from paddleocr.ppocr.data.imaug.random_crop_data import EastRandomCropData, RandomCropImgMask

from paddleocr.ppocr.data.imaug.make_border_map import MakeBorderMap
from paddleocr.ppocr.data.imaug.make_shrink_map import MakeShrinkMap

import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    f = open('data/icdar2015/text_localization/train_icdar2015_label.txt')
    line = f.readline()

    image_path, label = line.split('\t')
    image_path = os.path.join('data/icdar2015/text_localization', image_path)
    data = {}
    data['label'] = label
    with open(image_path, 'rb') as f:
        img = f.read()
        data['image'] = img

    img_decoder = DecodeImage(img_mode='BGR', channel_first=False)
    # data ['image']变成 720，1280，3
    data = img_decoder(data)

    label_encoder = DetLabelEncode()
    # data ['polys'] 变成一个数组形状为 [10,4,2]
    # 10 表示有10个文本区域   每个文本区域用上下左右四个点定位，每个点在二维平面用两个坐标表示
    data = label_encoder(data)

    # # 随机旋转 10度 放大0.5 到 3倍
    # iaaugment = IaaAugment(
    #     augmenter_args=[{'type': 'Fliplr', 'args': {'p': 0.5}}, {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
    #                     {'type': 'Resize', 'args': {'size': [0.5, 3]}}])
    #
    # data = iaaugment(data)
    #
    # # 再次进行翻转，而且可能会损失部分文本 图片大小缩放到 640*640
    # east_random_cropdata = EastRandomCropData(size=[640, 640], max_tries=50, keep_ratio=True)
    # data = east_random_cropdata(data)

    # 0.4表示文本框内缩比例  0.3 跟0.7 是控制文本框粗细的两个参数  获取阈值图
    make_border_map = MakeBorderMap(shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7)
    data = make_border_map(data)   # data 多出threhold_map 和threshold_mask 两个键

    # 0.4还是表示缩放比例 把原来的文本区域缩小到原来的40%
    make_shrink_map = MakeShrinkMap(shrink_ratio=0.4, min_text_size=8)
    data = make_shrink_map(data)


    # normalizer_image = NormalizeImage(scale=1 / 255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order='hwc')
    # # 归一化
    # data = normalizer_image(data)

    # channel 放到第一个维度
    to_chw_image = ToCHWImage()
    data = to_chw_image(data)

    threshold_map = data['threshold_mask']

    plt.figure(figsize=(10, 10))
    plt.imshow(threshold_map)
    plt.show()
    keep_keys = KeepKeys(keep_keys=['image', 'threshold_map', 'threshold_mask', 'shrink_map',
                        'shrink_mask'])
    data = keep_keys(data)