import torch
import torch.nn as nn
import unittest
import shutil

import pypaddle.deprecated
import pypaddle.sparse
import pypaddle.util
import networkx as nx
from pypaddle.learning import train, test

class DeepDACellNetworkTest(unittest.TestCase):
    def setUp(self):
        self.possible_dataset_roots = ['/media/data/set/mnist', 'data/set/mnist']
        self.batch_size = 100
        self.train_loader, self.test_loader, _, self.dataset_root = pypaddle.util.get_mnist_loaders(self.batch_size, self.possible_dataset_roots)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feature, labels = iter(self.train_loader).next()
        self.input_shape = feature.shape[1:]
        self.output_size = int(labels.shape[-1])

    def tearDown(self):
        if self.dataset_root is not self.possible_dataset_roots[0]:
            print('Deleting', self.dataset_root)
            shutil.rmtree(self.dataset_root)

    def test_develop(self):
        return

        # Arrange
        random_graph = nx.watts_strogatz_graph(30, 3, 0.8)
        structure = pypaddle.sparse.CachedLayeredGraph()
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

        # download and transform train dataset
        mnist_train_loader = self.train_loader
        mnist_test_loader = self.test_loader

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