import os
import argparse

import mindspore
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from resnet import resnet50
from wideresnet import wrn_50_2
from data_preprocess import get_rp2k_dataset

def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train rp2k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str, default='', help='the training data')
    parser.add_argument('--data_url', type=str, default='./data/all', help='the training data')
    parser.add_argument('--output_path', default='./checkpoints', type=str, help='the path model saved')
    parser.add_argument('--min_lr', default=1e-6, help='cos decay min lr')
    parser.add_argument('--max_lr', default=1e-3, help='con decay max lr')
    parser.add_argument('--epoch_size', default=75, help='no.epochs')
    parser.add_argument('--batch_size', default=128, help='batch size')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt

def train(args_opt):

    network = wrn_50_2(num_classes=2388, pretrained=False)
    lr = nn.cosine_decay_lr(min_lr = 1e-6, max_lr = 1e-3,
                            total_step = args_opt.batch_size * args_opt.epoch_size,
                            step_per_epoch = args_opt.batch_size,
                            decay_epoch = args_opt.epoch_size)
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")

    train_set = get_rp2k_dataset(args_opt.data_url + "/train", do_train=True)
    eval_set = get_rp2k_dataset(args_opt.data_url + "/test", do_train=False)

    steps_per_epoch = train_set.get_dataset_size()
    config_ck = CheckpointConfig(save_checkpoint_steps = steps_per_epoch, keep_checkpoint_max = 10)

    ckpt_callback = ModelCheckpoint(prefix = 'wrn_50_2_rp2k', directory = args_opt.output_path, config = config_ck)
    loss_callback = LossMonitor(steps_per_epoch)

    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc"})
    model.fit(args_opt.epoch_size, train_set, eval_set, callbacks = [ckpt_callback, loss_callback])