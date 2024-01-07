import torch
from torch import nn


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes,
                 include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.in_channel = 64

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=8, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  # ResNet-18/34会直接跳过该if语句（对于layer1来说）
            downsample = nn.Sequential(  # 下采样
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))  # 将特征矩阵的深度翻4倍，高和宽不变（对于layer1来说）

        layers = []
        layers.append(block(self.in_channel,  # 输入特征矩阵深度，64
                            channel,  # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):  # 从第二层开始都是实线残差结构
            layers.append(block(self.in_channel,  # 对于浅层一直是64，对于深层已经是64*4=256了
                                channel))  # 残差结构主分支上的第一层卷积的卷积核个数

        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)  # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回

    # 正向传播过程
    def forward(self, x):
        x = self.conv1(x)  # 8×8卷积层
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 3×3 max pool

        x = self.layer1(x)  # conv2_x所对应的一系列残差结构
        x = self.layer2(x)  # conv3_x所对应的一系列残差结构
        x = self.layer3(x)  # conv4_x所对应的一系列残差结构
        x = self.layer4(x)  # conv5_x所对应的一系列残差结构

        if self.include_top:
            x = self.avgpool(x)  # 平均池化下采样
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
