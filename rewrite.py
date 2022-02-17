import torch
import torch.nn as nn
import torch.nn.functional as F
def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    # 首先读取输入x的batch， channel， 长， 宽
    batch_size, channels, height, width = x.size()
    # 如果 channel数量不是group的整数倍，报错
    assert channels % groups == 0
    # 每组的channel个数
    channels_per_group = channels // groups
    # split into groups
    # 将输入x进行view处理，batch， 组数， 每组的channel数量， 长， 宽
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    # 对组数和每组的channel这两个维度进行转置
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    # 恢复 batch channel 长 宽
    x = x.view(batch_size, channels, height, width)
    return x

# ShuffleNet的模块A 步长为1的模块
class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        # 如果输入输出channel不一致，那么报错
        assert in_channels == out_channels
        # 如果输入channel不是group的倍数，那么报错
        assert in_channels % groups == 0
        # 如果输出channel不是4的倍数，那么报错
        assert out_channels % 4 == 0
        # 每一个bottleneck的channel个数
        bottleneck_channels = out_channels // 4
        # 如果bottleneck_channel不是group的倍数，那么报错
        assert bottleneck_channels % groups == 0
        self.groups = groups
        # 1、分组卷积 首先输入通道数量改为bottleneck通道数量，分组卷积， in_channels->in_channels//4
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                        kernel_size=1, groups=groups, stride=1)
        # 2、批归一化 batchnorm2d 通道数量为bottleneck的通道数
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # 3、深度可分离卷积 通道数量为bottleneck的通道数，分组数量也是bottleneck数量
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         kernel_size=3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        # 4、批归一化
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        # 5、 分组卷积 bottleneck通道数量改为out_channels的数量
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     kernel_size=1, stride=1, groups=groups)
        # 6、 批归一化
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 是一个残差连接
        # 首先进行一个分组卷积，in_channels分组卷积变为bottleneck_channels的数量
        out = self.group_conv1(x)
        # 之后进行批处理后激活函数操作
        out = F.relu(self.bn2(out))
        # 通道均匀打乱
        out = shuffle_channels(out, groups=self.groups)
        # 深度可分离卷积，group为bottleneck的数量
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        # 分组卷积，group为bottleneck数量恢复原有的通道数量
        out = self.group_conv5(out)
        out = self.bn6(out)
        # 和原有的通道进行相加
        out = F.relu(x + out)
        return out

# ShuffleNet的模块B 步长为1的模块
class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        # 为了合并通道，用concat，对输出的通道数量先减去输入通道数量
        out_channels -= in_channels
        # 如果输出通道数目不是4的倍数那么就报错
        assert out_channels % 4 == 0
        # bottleneck的通道数量就是 合并前 输出通道数量的1/4
        bottleneck_channels = out_channels // 4
        # 如果输入通道数目不是group的倍数那么就报错
        assert in_channels % groups == 0
        # 如果bottleneck通道数目不是group的倍数那么就报错
        assert bottleneck_channels % groups == 0
        self.groups = groups
        # 分组卷积，in_channels--> bottleneck_channels
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        # 批处理
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # 深度可分离卷积， 步长为2， bottleneck_channels -> bottleneck_channels 尺寸也缩小为原来的1/2*1/2
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        # 批处理
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        # 分组卷积，恢复原有通道数量 in_channels--> bottleneck_channels
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        # 批处理
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        # 大小变为1/2*1/2 通道数为目的的输出通道数量
        out = F.relu(torch.cat([x, out], dim=1))
        return out



class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        # (input_size-kernel+2padding)/stride + 1 = (7-3+2)/2+1=4 (6-3+2)/2+1=3
        self.conv1 = nn.Conv2d(in_channels, out_channels=24, kernel_size=3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
            [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
            [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        # batch, 3, input_size, input_size
        net = self.conv1(x)
        # batch, 24, input_size/2, input_size/2
        net = F.max_pool2d(net, 3, stride=2, padding=1)
        # batch, 24, input_size/4, input_size/4
        net = self.stage2(net)
        # batch, 240, input_size/8, input_size/8
        net = self.stage3(net)
        # batch, 480, input_size/16, input_size/16
        net = self.stage4(net)
        # batch, 960, 1, 1
        net = F.avg_pool2d(net, 7)
        net = net.view(net.size(0), -1)
        net = self.fc(net)
        logits = F.softmax(net)
        return logits

net = ShuffleNet()
print(net)
x = torch.rand(1, 3, 224, 224)# 112 56 28 7
print(net(x).size())