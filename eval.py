from wideresnet import wrn_50_2
from data_preprocess import get_rp2k_dataset
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore import load_checkpoint

def eval_wrn_50_2(epoch_size = 75, batch_size = 128, min_lr = 1e-6, max_lr = 1e-3):

    network = wrn_50_2()
    lr = nn.cosine_decay_lr(min_lr = min_lr, max_lr = max_lr,
                            total_step = batch_size * epoch_size,
                            step_per_epoch = batch_size,
                            decay_epoch = epoch_size)
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")

    eval_set = get_rp2k_dataset("./data/all/test", do_train=False)

    load_checkpoint("./checkpoints/wrn_50_2_ms.ckpt", net = network)

    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc"})
    acc = model.eval(eval_set)

    print("Accuracy of model is:{}%".format(acc['acc'] * 100))