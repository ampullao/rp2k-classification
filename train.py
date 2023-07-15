import mindspore
import mindspore.context as context
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import (
    CheckpointConfig,
    ModelCheckpoint,
    LossMonitor,
    TimeMonitor,
    SummaryCollector,
)

from src.model_utils.config import config
from src.wideresnet import wrn_50_2
from src.dataset import *

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

def train():
    network = wrn_50_2(num_classes=2388)
    lr = config.lr
    momentum = config.momentum
    opt = nn.Momentum(
        params=network.trainable_params(), learning_rate=lr, momentum=momentum
    )
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if config.resume_ckpt != "":
        param_dict = mindspore.load_checkpoint(config.resume_ckpt)
        mindspore.load_param_into_net(network, param_dict)
        mindspore.load_param_into_net(opt, param_dict)

    train_set = get_rp2k_dataset(config.data_path, do_train=True)

    steps_per_epoch = train_set.get_dataset_size()
    config_ck = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5
    )

    ckpt_callback = ModelCheckpoint(
        prefix="wrn_50_2_rp2k", directory=config.output, config=config_ck
    )
    loss_callback = LossMonitor(steps_per_epoch)
    time_callback = TimeMonitor()
    summary_collector = SummaryCollector(summary_dir=config.summary_base_dir, collect_freq=32)

    model = Model(network, loss_fn=loss_fn, optimizer=opt, metrics={"acc", "loss"})
    model.train(
        config.epoch_size,
        train_set,
        callbacks=[ckpt_callback, loss_callback, summary_collector, time_callback],
        dataset_sink_mode=False,
    )



if __name__ == "__main__":
    train()
