import paddle
import paddle.nn.functional as F
from paddle import nn

class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Spa_att(nn.Layer):
    def __init__(self):
        super(Spa_att, self).__init__()
        self.spatial = BasicConv(3, 1, 7, stride=1, padding=(kernel_size-1) // 2, bn=True, relu=False)
    def forward(self, x):
        x_out = self.spatial(x)
        scale = F.sigmoid(x_out)
        return x * scale+x

class Flatten(paddle.nn.Layer):
    def forward(self, x):
        return paddle.flatten(x, start_axis=1)

class Cha_att(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(Cha_att, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(lp_pool)


            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale