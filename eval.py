import os
import argparse
from wideresnet import wrn_50_2
from data_preprocess import get_rp2k_dataset
import mindspore
import mindspore.context as context
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore import load_checkpoint

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="eval rp2k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--checkpoint', type=str ,help='Model checkpoint file')
    parser.add_argument('--data_url', type=str, default='./data/all', help='the data path')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt

args = parse_args()

def eval_wrn_50_2(args):

    network = wrn_50_2()
    lr = 0.01
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")
    
    eval_set = get_rp2k_dataset(args.data_url + "/test", do_train=False)
    
    param_dict = mindspore.load_checkpoint("./checkpoints/" + args.checkpoint)
    mindspore.load_param_into_net(network, param_dict)
    
    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc"})
    
    print("start eval..................")
    acc = model.eval(eval_set, dataset_sink_mode = False)

    print("Accuracy of model is:{}%".format(acc['acc'] * 100))
    
if __name__ == "__main__":
    eval_wrn_50_2(args)