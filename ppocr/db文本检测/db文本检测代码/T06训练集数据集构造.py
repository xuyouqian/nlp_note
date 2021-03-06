import numpy as np
import os
import random
import paddle
import traceback
from paddle.io import Dataset,DataLoader
from paddle.io import DistributedBatchSampler,BatchSampler

from paddleocr.ppocr.data.imaug.operators import DecodeImage, NormalizeImage, ToCHWImage, KeepKeys
from paddleocr.ppocr.data.imaug.label_ops import DetLabelEncode

from paddleocr.ppocr.data.imaug.iaa_augment import IaaAugment
from paddleocr.ppocr.data.imaug.random_crop_data import EastRandomCropData, RandomCropImgMask

from paddleocr.ppocr.data.imaug.make_border_map import MakeBorderMap
from paddleocr.ppocr.data.imaug.make_shrink_map import MakeShrinkMap


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


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:2]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None or data['polys'].shape[1] != 4:
                continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


if __name__ == '__main__':
    config = {
        'Global': {'debug': False, 'use_gpu': False, 'epoch_num': 1200, 'log_smooth_window': 20, 'print_batch_step': 10,
                   'save_model_dir': './output/db_mv3/', 'save_epoch_step': 1200, 'eval_batch_step': [0, 2000],
                   'cal_metric_during_train': False,
                   'pretrained_model': './pretrain_models/MobileNetV3_large_x0_5_pretrained', 'checkpoints': None,
                   'save_inference_dir': None, 'use_visualdl': False, 'infer_img': 'doc/imgs_en/img_10.jpg',
                   'save_res_path': './output/det_db/predicts_db.txt', 'distributed': False},
        'Architecture': {'model_type': 'det', 'algorithm': 'DB', 'Transform': None,
                         'Backbone': {'name': 'MobileNetV3', 'scale': 0.5, 'model_name': 'large'},
                         'Neck': {'name': 'DBFPN', 'out_channels': 256}, 'Head': {'name': 'DBHead', 'k': 50}},
        'Loss': {'name': 'DBLoss', 'balance_loss': True, 'main_loss_type': 'DiceLoss', 'alpha': 5, 'beta': 10,
                 'ohem_ratio': 3},
        'Optimizer': {'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'lr': {'learning_rate': 0.001},
                      'regularizer': {'name': 'L2', 'factor': 0}},
        'PostProcess': {'name': 'DBPostProcess', 'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
                        'unclip_ratio': 1.5}, 'Metric': {'name': 'DetMetric', 'main_indicator': 'hmean'}, 'Train': {
            'dataset': {'name': 'SimpleDataSet', 'data_dir': 'data/icdar2015/text_localization/',
                        'label_file_list': ['data/icdar2015/text_localization/train_icdar2015_label.txt'],
                        'ratio_list': [1.0],
                        'transforms': [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                                       {'DetLabelEncode': None}, {'IaaAugment': {
                                'augmenter_args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                   {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                                                   {'type': 'Resize', 'args': {'size': [0.5, 3]}}]}}, {
                                           'EastRandomCropData': {'size': [640, 640], 'max_tries': 50,
                                                                  'keep_ratio': True}},
                                       {'MakeBorderMap': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}},
                                       {'MakeShrinkMap': {'shrink_ratio': 0.4, 'min_text_size': 8}}, {
                                           'NormalizeImage': {'scale': '1./255.', 'mean': [0.485, 0.456, 0.406],
                                                              'std': [0.229, 0.224, 0.225], 'order': 'hwc'}},
                                       {'ToCHWImage': None}, {'KeepKeys': {
                                'keep_keys': ['image', 'threshold_map', 'threshold_mask', 'shrink_map',
                                              'shrink_mask']}}]},
            'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 8,
                       'use_shared_memory': False}}, 'Eval': {
            'dataset': {'name': 'SimpleDataSet', 'data_dir': 'data/icdar2015/text_localization/',
                        'label_file_list': ['data/icdar2015/text_localization/test_icdar2015_label.txt'],
                        'transforms': [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                                       {'DetLabelEncode': None}, {'DetResizeForTest': {'image_shape': [736, 1280]}}, {
                                           'NormalizeImage': {'scale': '1./255.', 'mean': [0.485, 0.456, 0.406],
                                                              'std': [0.229, 0.224, 0.225], 'order': 'hwc'}},
                                       {'ToCHWImage': None},
                                       {'KeepKeys': {'keep_keys': ['image', 'shape', 'polys', 'ignore_tags']}}]},
            'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 1, 'num_workers': 8,
                       'use_shared_memory': False}}, 'profiler_options': None}
    import logging

    device = paddle.set_device('cpu')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    mode = 'Train'

    dataset = SimpleDataSet(config, mode, logger)

    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    if mode == "Train":
        # Distribute data to multiple cards
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    else:
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory)

    for i in data_loader():
        i


