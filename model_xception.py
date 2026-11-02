import torch.nn.functional as F
import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=False)
        rep = []

        filters = in_channels
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class XceptionTemporal(nn.Module):
    def __init__(self, num_classes=2, seq_length=4, pretrained_path=None):
        super(XceptionTemporal, self).__init__()
        self.seq_length = seq_length

        # 初始化Xception编码器
        self.encoder = Xception(num_classes=0)

        # 加载预训练权重
        if pretrained_path is not None:
            print(f"加载预训练权重: {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')

            # 如果预训练权重是完整模型，只提取encoder部分
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']

            # 过滤出Xception相关的权重
            encoder_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('encoder.') or k.startswith('features.') or k.startswith('conv') or k.startswith(
                        'bn') or k.startswith('block'):
                    # 移除可能的前缀
                    new_key = k.replace('encoder.', '').replace('features.', '')
                    encoder_dict[new_key] = v

            # 加载权重
            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"缺失的键: {len(missing_keys)}")
            print(f"意外的键: {len(unexpected_keys)}")

            # 不冻结Xception参数，继续训练
            print("Xception分支已加载预训练权重，将继续训练")

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 输出2个类别
        )

    def forward(self, x):
        batch_size, vectors, seq_len, C, H, W = x.shape
        x = x.view(batch_size * vectors * seq_len, C, H, W)

        # Xception编码器前向传播
        xception_features = self.encoder(x)

        # 强制特征维度为2048
        if xception_features.numel() == 1 or xception_features.dim() != 2:
            xception_features = torch.zeros(batch_size * vectors * seq_len, 2048, device=x.device)
        elif xception_features.size(-1) != 2048:
            xception_features = torch.nn.functional.pad(xception_features, (0, 2048 - xception_features.size(-1)))

        # 重塑特征
        features = xception_features.view(batch_size * vectors, seq_len, -1)

        # 对序列维度进行平均池化
        features = features.mean(dim=1)  # [batch_size * vectors, 2048]

        # 分类
        cls_output = self.classifier(features)

        # 对vectors维度进行平均
        final_output = cls_output.view(batch_size, vectors, -1).mean(1)
        return final_output