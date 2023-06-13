import os
import mindspore.common.dtype as mtype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms. c_transforms as C2

def create_dateset(dataset_path, do_train, batch_size = 128):
    