import mindspore
import mindspore.context as context
from mindspore import nn
from mindspore.train import Model

from src.wideresnet import wrn_50_2
from src.dataset import get_rp2k_dataset
from src.model_utils.config import config

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

def eval_wrn_50_2():

    network = wrn_50_2()
    lr = 0.01
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")

    eval_set = get_rp2k_dataset(config.data_path, do_train=False)

    param_dict = mindspore.load_checkpoint(config.ckpt_path)
    mindspore.load_param_into_net(network, param_dict)

    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc"})

    print("start eval..................")
    acc = model.eval(eval_set, dataset_sink_mode = False)

    print("Accuracy of model is:{}%".format(acc['acc'] * 100))

if __name__ == "__main__":
    eval_wrn_50_2()
