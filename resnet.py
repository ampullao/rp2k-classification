from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net

class ResNetBlock(nn.Cell):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride = 1):
        super().__init__()
        # first layer
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 1)
        self.norm1 = nn.BatchNorm2d(out_channel)

        # second layer 
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = stride)
        self.norm2 = nn.BatchNorm2d(out_channel)

        # third layer
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size = 1)
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.down_sample = nn.SequentialCell([
            nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size = 1, stride = 2),
            nn.BatchNorm2d(out_channel * self.expansion)
            ])

    def construct(self, x):
        identity = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out
    
def make_layer(last_out_channel, channel, block_nums, stride = 1):
    layers = []
    layers.append(ResNetBlock(last_out_channel, channel, stride = stride))

    in_channel = channel * ResNetBlock.expansion
    for _ in (1, block_nums):
        layers.append(ResNetBlock(in_channel, channel))
    
    return nn.SequentialCell(layers)

class ResNet(nn.Cell):
    def __init__(self, layer_nums, num_classes, input_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2)
        self.norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, pad_mode = 'same')

        self.layer1 = make_layer(64, 64, layer_nums[0])
        self.layer2 = make_layer(64 * ResNetBlock.expansion, 128, layer_nums[1], stride = 2)
        self.layer3 = make_layer(128 * ResNetBlock.expansion, 256, layer_nums[2], stride = 2)
        self.layer4 = make_layer(256 * ResNetBlock.expansion, 512, layer_nums[3], stride = 2)

        self.avg_pool = nn.AvgPool2d()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels = input_channel, out_channels = num_classes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

def resnet50(num_classes = 2388, pretrained = False):
    resnet50_ckpt = "../LoadPretrainedModel/resnet50_224_new.ckpt"
    layer_nums = [3, 4, 6, 3]
    model = ResNet(layer_nums, num_classes, 2048)

    if pretrained:
        param_dict = load_checkpoint(resnet50_ckpt)
        load_param_into_net(model, param_dict)
    
    return model