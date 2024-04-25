from torch import nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
