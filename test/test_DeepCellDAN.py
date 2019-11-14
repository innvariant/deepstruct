import math
import torch
import torch.nn as nn
import unittest

import pypaddle.deprecated
import pypaddle.sparse
import pypaddle.util
from pypaddle.learning import train, test

class DeepCellDANTest(unittest.TestCase):
    def test_development(self):
        return

        train_loader, test_loader, val_loader = pypaddle.util.get_cifar10_loaders()

        class ReductionCell(nn.Module):
            def __init__(self, input_channels, output_channels):
                super().__init__()

                self.conv_reduce = nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=2, stride=2)
                self.act = nn.ReLU()
                self.batch_norm = nn.BatchNorm2d(output_channels)

            def forward(self, input):
                return self.batch_norm(self.act(self.conv_reduce(input)))

        class Foo(nn.Module):
            def __init__(self, num_classes: int, input_channels: int, input_size: int):
                super(Foo, self).__init__()

                self._num_classes = num_classes

                reduced_channels = 100
                reduction_steps = max(math.floor(math.log2(input_size))-5, 1)
                input_convs = [ReductionCell(input_channels, reduced_channels)]
                for step in range(1, reduction_steps):
                    input_convs.append(ReductionCell(reduced_channels, reduced_channels))
                self._input_convs = nn.ModuleList(input_convs)

                self.conv1 = nn.Conv2d(reduced_channels, 100, (1, 1))
                self.bn1 = nn.BatchNorm2d(100)
                self.fc1 = nn.Linear(100, self._num_classes)

            def forward(self, input):
                y = input
                for conv in self._input_convs:
                    y = conv(y)

                y = self.conv1(y)
                y = self.bn1(y) # [B, X, N, M]
                y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1)) # [B, X, 1, 1]
                y = y.view(y.size(0), -1) # [B, X]
                return self.fc1(y)  # [B, _num_classes]

        model = Foo(10, 3, 32)
        print(model)
        print([p.shape for p in model.parameters()])

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(50):
            print('Epoch', epoch)
            model.train()
            print(train(train_loader, model, optimizer, loss_func, device))
            model.eval()
            print(test(test_loader, model, device))

        print('Done')
        print('Test:')
        model.eval()
        print(test(test_loader, model, device))

        print('Validation:')
        model.eval()
        print(test(val_loader, model, device))