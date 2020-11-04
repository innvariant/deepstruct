import torch.nn as nn
import torch.nn.functional as fun


class MaxPoolLayers(nn.Module):
    model_name = "MaxPoolLayers"

    def __init__(self):
        super(MaxPoolLayers, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.max_pool(x)
        return out


class ConvLayers(nn.Module):
    model_name = "ConvolutionLayers"

    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, 1)

    def forward(self, x):
        out = fun.relu(self.conv1(x))
        return out


class ConvMaxPoolLayers(nn.Module):
    model_name = "ConvMaxPoolLayers"

    def __init__(self):
        super(ConvMaxPoolLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = fun.relu(self.conv1(x))
        out = self.max_pool(out)
        return out


class Conv_Padding_Layers(nn.Module):
    model_name = "Conv_Padding_Layers"

    def __init__(self):
        super(Conv_Padding_Layers, self).__init__()
        self._conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = fun.relu(self._conv1(x))
        return out


class Conv_Padding_Multi_In(nn.Module):
    model_name = "Conv_Padding_Multi_In"

    def __init__(self):
        super(Conv_Padding_Multi_In, self).__init__()
        self._conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = fun.relu(self._conv1(x))
        return out


class LinearLayer(nn.Module):
    model_name = "LinearLayer"

    def __init__(self):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.Linear(6, 3)

    def forward(self, x):
        out = fun.relu(self.fc1(x))
        return out
