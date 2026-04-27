import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, dropoutRate=0.5, F1=8, D=2):
        super(EEGNet, self).__init__()

        F2 = F1 * D

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, Samples // 8), padding=(0, (Samples // 8)//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.sep_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, (Samples // 8)//2), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        self.fc = nn.LazyLinear(nb_classes)

    def forward(self, x):
        # input: (batch, chans, samples, 1) → convert
        x = x.permute(0, 3, 1, 2)  # → (batch, 1, chans, samples)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x  # logits (NO softmax)