import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

# 定义残差块
class Bottleneck(nn.Cell):
    def __init__(self, in_channels, out_channels, k, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // 4 * k
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if stride != 1 or in_channels != out_channels:
        # 如果输入输出维度不一致，需要用一个1x1卷积进行匹配
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
                ])
        else:
        # 否则直接相加
            self.shortcut = None

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out

# 定义WRN-50-2网络
class WRN_50_2(nn.Cell):
    def __init__(self, num_classes=2388):
        super(WRN_50_2, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 四个卷积组，每个卷积组由若干个残差块组成
        # 每个卷积组的第一个残差块的步长为2，其他为1
        # 每个卷积组的输出通道数为上一个卷积组的两倍
        # WRN-50-2的宽度系数k为2，即通道数翻倍
        # WRN-50-2的残差块数为[3, 4, 6, 3]
        k = 2
        init_block_channels = 64
        block_nums = [3, 4, 6, 3]
        channels_per_layer = [256, 512, 1024, 2048]

        # 第一个卷积组，输出通道数为64×k，步长为1
        self.layer1 = self._make_layer(init_block_channels, channels_per_layer[0], k, block_nums[0], stride = 1)

        # 第二个卷积组，输出通道数为128×k，步长为2
        self.layer2 = self._make_layer(channels_per_layer[0], channels_per_layer[1], k, block_nums[1], stride = 2)

        # 第三个卷积组，输出通道数为256×k，步长为2
        self.layer3 = self._make_layer(channels_per_layer[1], channels_per_layer[2], k, block_nums[2], stride = 2)

        # 第四个卷积组，输出通道数为512×k，步长为2
        self.layer4 = self._make_layer(channels_per_layer[2], channels_per_layer[3], k, block_nums[3], stride = 2)

        # 平均池化层
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层
        self.fc = nn.Dense(channels_per_layer[3], num_classes)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, last_out_channel, channel, k, block_nums, stride = 1):
        layers = []
        layers.append(Bottleneck(last_out_channel, channel, k, stride = stride))

        for _ in (1, block_nums):
            layers.append(Bottleneck(channel, channel, k))
    
        return nn.SequentialCell(layers)

def wrn_50_2(num_classes = 2388, pretrained = False):
    wrn_50_2_ckpt = "../LoadPretrainedModel/resnet50_224_new.ckpt"
    model = WRN_50_2()

    if pretrained:
        param_dict = load_checkpoint(wrn_50_2_ckpt)
        load_param_into_net(model, param_dict)
    
    return model
