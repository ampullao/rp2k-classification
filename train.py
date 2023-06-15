import mindspore
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from resnet import resnet50
from wideresnet import wrn_50_2
from data_preprocess import get_rp2k_dataset

def train(epoch_size = 75, batch_size = 128, min_lr = 1e-6, max_lr = 1e-3, pretrained = False):

    network = wrn_50_2(pretrained=pretrained)
    lr = nn.cosine_decay_lr(min_lr = min_lr, max_lr = max_lr,
                            total_step = batch_size * epoch_size,
                            step_per_epoch = batch_size,
                            decay_epoch = epoch_size)
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")

    train_set = get_rp2k_dataset("./data/all/train", do_train=True)
    eval_set = get_rp2k_dataset("./data/all/test", do_train=False)

    steps_per_epoch = train_set.get_dataset_size()
    config = CheckpointConfig(save_checkpoint_steps = steps_per_epoch)

    ckpt_callback = ModelCheckpoint(prefix = 'wrn_50_2_rp2k', directory = './checkpoints', config = config)
    loss_callback = LossMonitor(steps_per_epoch)

    model = Model(network, loss_fn = loss_fn, optimizer = opt, metrics = {"acc"})
    model.fit(epoch_size, train_set, eval_set, callbacks = [ckpt_callback, loss_callback])