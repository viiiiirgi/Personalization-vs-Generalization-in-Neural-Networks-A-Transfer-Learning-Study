import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # remove padding effect
        out = self.bn1(out)
        out = F.elu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = F.elu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        return out + res

class TCN(nn.Module):
    def __init__(self, nb_classes, Chans, Samples,
                 n_filters=16, levels=3, kernel_size=7, dropout=0.5):
        super().__init__()

        self.Chans = Chans

        # first projection: from channels → feature space
        self.input_proj = nn.Conv1d(Chans, n_filters, kernel_size=1)

        # temporal blocks with increasing dilation
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    n_filters,
                    n_filters,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # global pooling (important for variable length)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(n_filters, nb_classes)

    def forward(self, x):
        # input: (batch, chans, samples, 1)
        x = x.squeeze(-1)  # → (batch, chans, samples)

        # TCN expects (batch, channels, time)
        x = self.input_proj(x)

        x = self.tcn(x)

        x = self.pool(x).squeeze(-1)

        x = self.fc(x)

        return x