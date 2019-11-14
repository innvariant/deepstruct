import os
import numpy as np
import torch
import torch.nn as nn
import unittest

from torch.utils.data import SubsetRandomSampler

import paddle.deprecated
import paddle.sparse
import paddle.util
import networkx as nx
import torchvision
from paddle.learning import train, test
from torchvision.transforms import transforms

class DeepDACellNetworkTest(unittest.TestCase):
    def test_develop(self):
        return

        # Arrange
        random_graph = nx.watts_strogatz_graph(30, 3, 0.8)
        structure = paddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)

        # Depthwise Separable Convolutions
        class SeparableConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
                super(SeparableConv2d, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
                self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

            def forward(self, x):
                x = self.conv1(x)
                x = self.pointwise(x)
                return x

        class CustomCell(nn.Module):
            def __init__(self, in_degree, in_channel, out_channel, stride):
                super(CustomCell, self).__init__()
                self.single = (in_degree == 1)
                if not self.single:
                    self.agg_weight = nn.Parameter(torch.zeros(in_degree, requires_grad=True))
                self.conv = SeparableConv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
                self.bn = nn.BatchNorm2d(out_channel)
                self.act = nn.ReLU()
                self.num_out_channel = out_channel

            def forward(self, input):
                # input: [B, C_in, N, M, in_degree]
                if self.single:
                    input = input.squeeze(-1)
                else:
                    input = torch.matmul(input, torch.sigmoid(self.agg_weight))  # [B, C_in, N, M]
                input = self.conv(self.act(input))  # [B, C_out, N, M]
                input = self.bn(input)  # [B, C_out, N, M]
                return input

        class SkipMap(nn.Module):
            def __init__(self, in_channel, out_channel):
                super(SkipMap, self).__init__()
                #self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
                self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
                self.bn = nn.BatchNorm2d(out_channel)
                self.act = nn.ReLU()

            def forward(self, input):
                # input: [B, C_in, N, M]
                input = self.conv(self.act(input))  # [B, C_out, N, M]
                input = self.bn(input)  # [B, C_out, N, M]
                return input

        def layer_channel_size(layer: int):
            # MNIST: input layer channel size 1
            # CIFAR: input layer channel size 3
            return 1 if layer is 0 else 40 if layer < 4 else 80 if layer < 8 else 160

        def construct_cell(is_input, is_output, in_degree, out_degree, layer):
            in_channel = layer_channel_size(0 if is_input else layer)
            in_degree = max(1, in_degree)  # shouldn't be zero
            out_channel = layer_channel_size(layer+1)
            cell = CustomCell(in_degree, in_channel, out_channel, 1)
            cell.layer = layer
            return cell

        model = paddle.deprecated.DeepDACellNetwork(10, construct_cell, layer_channel_size, SkipMap, structure)
        print(model)
        print([p.shape for p in model.parameters()])

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Trainable params', count_parameters(model))
        print('Total params', sum(p.numel() for p in model.parameters()))
        print('Non-trainable params', sum(p.numel() for p in model.parameters() if not p.requires_grad))

        data_base_path = '/home/julian/data'

        batch_size = 100

        custom_transform = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
        ])
        # download and transform train dataset
        mnist_train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(os.path.join(data_base_path, 'set/mnist/'),
                                       download=True,
                                       train=True,
                                       transform=custom_transform),
            batch_size=batch_size,
            shuffle=True)

        # download and transform test dataset
        mnist_test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(os.path.join(data_base_path, 'set/mnist/'),
                                       download=True,
                                       train=False,
                                       transform=custom_transform),
            batch_size=batch_size,
            shuffle=True)

        class ReshapeTransform:
            def __init__(self, new_size):
                self.new_size = new_size

            def __call__(self, img):
                return torch.reshape(img, self.new_size)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ReshapeTransform((3, 32, 32))])

        dataset_root = '/media/data/set/cifar10'
        if not os.path.exists(dataset_root):
            os.makedirs(dataset_root)
        train_set = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
        # Training
        n_training_samples = 20000
        train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
        # Validation
        n_val_samples = 5000
        val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
        # Test
        n_test_samples = 5000
        test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
        batch_size = 100
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=2)
        val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=2)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

        #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(250):
            print('Epoch', epoch)
            model.train()
            print(train(mnist_train_loader, model, optimizer, loss_func, device))

            model.eval()
            print(test(mnist_test_loader, model, device))

        print('Done')
        print('Test:')
        model.eval()
        print(test(mnist_test_loader, model, device))