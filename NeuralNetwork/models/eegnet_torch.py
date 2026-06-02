import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64, F1=4, D=2):
        super(EEGNet, self).__init__()

        F2 = F1 * D

        ## Block 1 Temporal and spatial filtering CONV2D
        # (1st layer) learns temporal patterns (finds frequencies that are relevant)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)

        self.bn1 = nn.BatchNorm2d(F1) #normalizes the data

        # (2nd layer) learns spatial patterns across electrodes. #DEPTHWISECONV2D
        #  Each temporal filter gets its own spatial filter. (which parts of the brain are communicating)
        self.depthwise = nn.Conv2d(F1, F2, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.pool1 = nn.AvgPool2d((1, 4)) # downsamples the data, keeping the important features while reducing noise and computational load
        self.dropout1 = nn.Dropout(dropoutRate) # prevents overfitting: randomly disables neurons

        # Block 2 SEPARABLECONV2D
        #separable convolution (depthwise temporal filtering and then pointwise mixing)
        self.sep_depthwise = nn.Conv2d(F2, F2, (1, 16), groups=F2, padding=(0, kernLength // 2), bias=False)# Per channel temporal filtering: applies filter independently to each channel (no mix information between channels)
        self.sep_pointwise = nn.Conv2d(F2, F2, (1, 1), bias=False) # mixes channels together and combines features
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
        x = F.elu(x) # non linear activation function that allows the network to learn complex patterns
        x = self.pool1(x)
        x = self.dropout1(x)

        #x = self.sep_conv(x)
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)

        x = self.bn3(x)
        x = F.elu(x) 
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1) # flattens the 2d feature maps into a 1d vector
        x = self.fc(x)

        return x 