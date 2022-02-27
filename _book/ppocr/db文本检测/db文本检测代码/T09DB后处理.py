from ppocr.data.imaug.operators import DecodeImage, NormalizeImage, ToCHWImage, KeepKeys, DetResizeForTest

from ppocr.data import transform
from ppocr.postprocess.db_postprocess import DBPostProcess
from T07_create_base_model import BaseModel
import json
import numpy as np
import cv2
import six
import os

import paddle


def load_model(model, path):
    """
    根据path提供的模型数据加载入模型
    :param model:
    :param path:
    :return:
    """
    # load params from trained model
    params = paddle.load(path)
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key not in params:
            print("{} not in loaded params {} !".format(
                key, params.keys()))
            continue
        pre_value = params[key]
        if list(value.shape) == list(pre_value.shape):
            new_state_dict[key] = pre_value
        else:
            print(
                "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
    model.set_state_dict(new_state_dict)


class DecodeImage(object):
    """ decode image """

    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        # 1. 图像解码
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)

        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))
        # 2. 解码后的图像放在字典中
        data['image'] = img.astype('float32')
        return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def draw_det_res(dt_boxes, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)

if __name__ == '__main__':

    process_config = {'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000, 'unclip_ratio': 1.5}

    # db 后处理过程
    post_process_class = DBPostProcess(**process_config)
    base_model = BaseModel()
    load_model(base_model, r'D:\code\ocr_pdpd\PaddleOCR\data\det_mv3_db_v2.0_train\best_accuracy.pdparams')

    transforms = [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                  {'DetResizeForTest': {'image_shape': [736, 1280]}},
                  {'NormalizeImage': {'scale': '1./255.', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                      'order': 'hwc'}},
                  {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['image', 'shape']}}]



    ops = create_operators(transforms)

    with open(r'D:\code\ocr_pdpd\PaddleOCR\data\img_12.jpg', 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)

    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    images = paddle.to_tensor(images)
    base_model.eval()
    preds = base_model(images)  # [1,1,736,1280]

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(preds['maps'][0][0])
    # plt.show()

    # 这里面是很多框的信息
    post_result = post_process_class(preds, shape_list)

    src_img = cv2.imread(r'D:\code\ocr_pdpd\PaddleOCR\data\img_12.jpg')

    dt_boxes_json = []
    # parser boxes if post_result is dict
    if isinstance(post_result, dict):
        det_box_json = {}
        for k in post_result.keys():
            boxes = post_result[k][0]['points']
            dt_boxes_list = []
            for box in boxes:
                tmp_json = {"transcription": ""}
                tmp_json['points'] = box.tolist()
                dt_boxes_list.append(tmp_json)
            det_box_json[k] = dt_boxes_list
            save_det_path = ''
            draw_det_res(boxes,  src_img, 'res', save_det_path)
    else:
        boxes = post_result[0]['points']
        dt_boxes_json = []
        # write result
        for box in boxes:
            tmp_json = {"transcription": ""}
            tmp_json['points'] = box.tolist()
            dt_boxes_json.append(tmp_json)
        save_det_path = 'savepath'
        draw_det_res(boxes,  src_img, 'file.jpg', save_det_path)

