# from download import download
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.common.dtype as mstype

# url = "https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/" \
    #   "rp2k_dataset.zip"
# path = download(url, "./data", kind = "zip", replace = True)

def get_rp2k_dataset(dataset_path, do_train, batch_size = 128):
    # mean and std of Imagenet
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # get data from directory
    if do_train:
        dataset = ds.ImageFolderDataset(dataset_dir = dataset_path, shuffle = True)
        # datapipe
        trans = [vision.RandomCropDecodeResize(224),
                 vision.Normalize(mean = mean, std = std),
                 vision.HWC2CHW()]
    else:
        dataset = ds.ImageFolderDataset(dataset_dir = dataset_path)
        # datapipe
        trans = [vision.Decode(),
                 vision.Resize(256),
                 vision.CenterCrop(224),
                 vision.Normalize(mean = mean, std = std),
                 vision.HWC2CHW()]
    
    dataset = dataset.map(input_columns='image', operations = trans)
    dataset = dataset.map(input_columns='label', operations = transforms.TypeCast(mstype.int32))
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)

    return dataset
