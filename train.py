from resnet import resnet50


def resnet50_train():

    network = resnet50()

    epoch_size = 75
    batch_size = 128
    lr = nn.cosine_decay_lr(min_lr = 1e-6, max_lr = 1e-3,
                            total_step = batch_size * epoch_size,
                            step_per_epoch = batch_size,
                            decay_epoch = epoch_size)
    opt = nn.Adam(params = network.trainable_params(), learning_rate = lr)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = "mean")
