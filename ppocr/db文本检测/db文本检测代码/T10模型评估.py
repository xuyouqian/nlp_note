# 创建模型并加载
from T07_create_base_model import BaseModel
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


# 准备验证集的数据集
from paddleocr.ppocr.data.simple_dataset import SimpleDataSet
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 验证准确率的类
from ppocr.metrics.det_metric import DetMetric

# 数据后处理的类
from ppocr.postprocess.db_postprocess import DBPostProcess

# 验证函数
from tqdm import tqdm
import platform
import time


def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False):
    model.eval()
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()
            if model_type == 'table' or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type == "kie":
                preds = model(batch)
            else:
                preds = model(images)
            batch = [item.numpy() for item in batch]
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ['table', 'kie']:
                eval_class(preds, batch)
            else:
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


if __name__ == '__main__':
    # 创建模型并加载
    base_model = BaseModel()
    load_model(base_model, r'data\det_mv3_db_v2.0_train\best_accuracy.pdparams')
    base_model.eval()

    # 准备验证集的数据集

    config = {
        'Global': {
            'debug': False,
            'use_gpu': False,
            'epoch_num': 1200,
            'log_smooth_window': 20,
            'print_batch_step': 10,
            'save_model_dir': './output/db_mv3/',
            'save_epoch_step': 1200,
            'eval_batch_step': [
                0,
                1
            ],
            'cal_metric_during_train': False,
            'pretrained_model': './pretrain_models/MobileNetV3_large_x0_5_pretrained',
            'checkpoints': None,
            'save_inference_dir': None,
            'use_visualdl': False,
            'infer_img': 'doc/imgs_en/img_10.jpg',
            'save_res_path': './output/det_db/predicts_db.txt',
            'distributed': False
        },
        'Architecture': {
            'model_type': 'det',
            'algorithm': 'DB',
            'Transform': None,
            'Backbone': {
                'name': 'MobileNetV3',
                'scale': 0.5,
                'model_name': 'large'
            },
            'Neck': {
                'name': 'DBFPN',
                'out_channels': 256
            },
            'Head': {
                'name': 'DBHead',
                'k': 50
            }
        },
        'Loss': {
            'name': 'DBLoss',
            'balance_loss': True,
            'main_loss_type': 'DiceLoss',
            'alpha': 5,
            'beta': 10,
            'ohem_ratio': 3
        },
        'Optimizer': {
            'name': 'Adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'lr': {
                'learning_rate': 0.001
            },
            'regularizer': {
                'name': 'L2',
                'factor': 0
            }
        },
        'PostProcess': {
            'name': 'DBPostProcess',
            'thresh': 0.3,
            'box_thresh': 0.6,
            'max_candidates': 1000,
            'unclip_ratio': 1.5
        },
        'Metric': {
            'name': 'DetMetric',
            'main_indicator': 'hmean'
        },
        'Train': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': '../data/icdar2015/text_localization/',
                'label_file_list': [
                    '../data/icdar2015/text_localization/train_icdar2015_label.txt'
                ],
                'ratio_list': [
                    1.0
                ],
                'transforms': [
                    {
                        'DecodeImage': {
                            'img_mode': 'BGR',
                            'channel_first': False
                        }
                    },
                    {
                        'DetLabelEncode': None
                    },
                    {
                        'IaaAugment': {
                            'augmenter_args': [
                                {
                                    'type': 'Fliplr',
                                    'args': {
                                        'p': 0.5
                                    }
                                },
                                {
                                    'type': 'Affine',
                                    'args': {
                                        'rotate': [
                                            -10,
                                            10
                                        ]
                                    }
                                },
                                {
                                    'type': 'Resize',
                                    'args': {
                                        'size': [
                                            0.5,
                                            3
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                    {
                        'EastRandomCropData': {
                            'size': [
                                640,
                                640
                            ],
                            'max_tries': 50,
                            'keep_ratio': True
                        }
                    },
                    {
                        'MakeBorderMap': {
                            'shrink_ratio': 0.4,
                            'thresh_min': 0.3,
                            'thresh_max': 0.7
                        }
                    },
                    {
                        'MakeShrinkMap': {
                            'shrink_ratio': 0.4,
                            'min_text_size': 8
                        }
                    },
                    {
                        'NormalizeImage': {
                            'scale': '1./255.',
                            'mean': [
                                0.485,
                                0.456,
                                0.406
                            ],
                            'std': [
                                0.229,
                                0.224,
                                0.225
                            ],
                            'order': 'hwc'
                        }
                    },
                    {
                        'ToCHWImage': None
                    },
                    {
                        'KeepKeys': {
                            'keep_keys': [
                                'image',
                                'threshold_map',
                                'threshold_mask',
                                'shrink_map',
                                'shrink_mask'
                            ]
                        }
                    }
                ]
            },
            'loader': {
                'shuffle': True,
                'drop_last': False,
                'batch_size_per_card': 16,
                'num_workers': 8,
                'use_shared_memory': False
            }
        },
        'Eval': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': 'data/icdar2015/text_localization/',
                'label_file_list': [
                    'data/icdar2015/text_localization/test_icdar2015_label.txt'
                ],
                'transforms': [
                    {
                        'DecodeImage': {
                            'img_mode': 'BGR',
                            'channel_first': False
                        }
                    },
                    {
                        'DetLabelEncode': None
                    },
                    {
                        'DetResizeForTest': {
                            'image_shape': [
                                736,
                                1280
                            ]
                        }
                    },
                    {
                        'NormalizeImage': {
                            'scale': '1./255.',
                            'mean': [
                                0.485,
                                0.456,
                                0.406
                            ],
                            'std': [
                                0.229,
                                0.224,
                                0.225
                            ],
                            'order': 'hwc'
                        }
                    },
                    {
                        'ToCHWImage': None
                    },
                    {
                        'KeepKeys': {
                            'keep_keys': [
                                'image',
                                'shape',
                                'polys',
                                'ignore_tags'
                            ]
                        }
                    }
                ]
            },
            'loader': {
                'shuffle': False,
                'drop_last': False,
                'batch_size_per_card': 1,
                'num_workers': 8,
                'use_shared_memory': False
            }
        },
        'profiler_options': None
    }

    eval_dataset = SimpleDataSet(config, 'Eval', logger, seed=None)

    # 验证准确率的lei
    eval_class = DetMetric(main_indicator='hmean')

    # db属于后处理
    process_config = {'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000, 'unclip_ratio': 1.5}
    post_process_class = DBPostProcess(**process_config)

    batch_sampler = BatchSampler(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False)

    data_loader = DataLoader(
        dataset=eval_dataset,
        batch_sampler=batch_sampler,
        return_list=True, )

    res = eval(base_model, data_loader, post_process_class, eval_class, model_type='det')
    print(res)
    '''

    {'precision': 0.7730749617542071, 'recall': 0.7306024096385542, 'hmean': 0.7512388503468781, 'fps': 0.5552333503751502}

    '''