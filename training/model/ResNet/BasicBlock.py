from torch import nn


class BasicBlock(nn.Module):
    expansion = 1   # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1

    # 定义初始函数
    # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):   # downsample对应虚线残差结构捷径中的1×1卷积
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用bn层时不使用bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 实/虚线残差结构主分支中第二层stride都为1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample   # 默认是None

# 定义正向传播过程
    def forward(self, x):
        identity = x   # 捷径分支的输出值
        if self.downsample is not None:   # 对应虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)   # 这里不经过relu激活函数

        out += identity
        out = self.relu(out)

        return out