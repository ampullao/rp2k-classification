import os
import argparse

import mindspore
import mindspore.context as context
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, SummaryCollector
from wideresnet import wrn_50_2
from data_preprocess import *

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train rp2k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--data_url', type=str, default='./data/all', help='the training data')
    parser.add_argument('--output_path', default='./checkpoints', type=str, help='the path model saved')
    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--momentum', default=0.9, help='momentum')
    parser.add_argument('--epoch_size', default=75, type=int, help='no.epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--load_ckpt', default='', help='load pretrained weight')
    parser.add_argument('--sum_dir', default='./summary_dir', help='summary dir')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt

args = parse_args()

def train(args_opt):

    network = wrn_50_2(num_classes=2388)
    lr = args_opt.lr
    momentum = args_opt.momentum
    opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=momentum)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")
    if args_opt.load_ckpt != "":
        param_dict = mindspore.load_checkpoint("./checkpoints/" + args_opt.load_ckpt)
        mindspore.load_param_into_net(network, param_dict)
        mindspore.load_param_into_net(opt, param_dict)

    train_set = get_rp2k_dataset(args_opt.data_url + "/train", do_train=True)
    eval_set = get_rp2k_dataset(args_opt.data_url + "/test", do_train=False)

    steps_per_epoch = train_set.get_dataset_size()
    config_ck = CheckpointConfig(save_checkpoint_steps = steps_per_epoch, keep_checkpoint_max = 5)

    ckpt_callback = ModelCheckpoint(prefix = 'wrn_50_2_rp2k', directory = args_opt.output_path, config = config_ck)
    loss_callback = LossMonitor(steps_per_epoch)
    time_callback = TimeMonitor()
    summary_collector = SummaryCollector(summary_dir=args_opt.sum_dir, collect_freq=32)

    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc", "loss"})
    model.train(args_opt.epoch_size, train_set, callbacks = [ckpt_callback, loss_callback, summary_collector, time_callback], dataset_sink_mode = False)

    metrics = model.eval(eval_set, dataset_sink_mode = False)
    print("Metrics:", metrics)
    
if __name__ == "__main__":
    train(args)