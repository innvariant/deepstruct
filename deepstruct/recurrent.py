import math

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import init


class BaseLayerModule(nn.Module):
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self._hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def set_i2h_mask(self, mask):
        self._mask_i2h = Variable(mask)

    def set_h2h_mask(self, mask):
        self._mask_h2h = Variable(mask)


class BaseMaskModule(nn.Module):
    def apply_mask(self, threshold=0.0, i2h=False, h2h=False):
        """
        :param threshold: Amount of pruning to apply. Default '0.0'
        :param i2h: If True, then Input-to-Hidden layers will be masked. Default 'False'
        :param h2h: If True, then Hidden-to-Hidden layers will be masked. Default 'False'
        :type threshold: float
        :type i2h: bool
        :type h2h: bool
        """
        if not i2h and not h2h:
            return

        masks = self.__get_masks(threshold, i2h, h2h)
        for lay_idx, layer in enumerate(self._recurrent_layers):
            if i2h:
                layer.set_i2h_mask(masks[lay_idx][0])
            if h2h:
                layer.set_h2h_mask(masks[lay_idx][-1])

    def __get_masks(self, threshold, i2h, h2h):
        key = "" if i2h and h2h else "ih" if i2h else "hh" if h2h else None

        masks = {}
        for lay_idx, layer in enumerate(self._recurrent_layers):
            masks[lay_idx] = []
            for param, data in layer.named_parameters():
                if "bias" not in param and key in param:
                    mask = torch.ones(data.shape, dtype=torch.bool, device=data.device)
                    mask[torch.where(abs(data) < threshold)] = False
                    masks[lay_idx].append(mask)

        return masks


class MaskedRecurrentLayer(BaseLayerModule):
    """
    Base class for layer initialization for Vanilla RNN.

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        nonlinearity: Can be either 'tanh', 'relu'
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity=torch.nn.Tanh(),
        batch_first: bool = False,
    ):
        super(MaskedRecurrentLayer, self).__init__()

        assert input_size > 0
        assert hidden_size > 0
        assert callable(nonlinearity)

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._nonlinearity = nonlinearity
        self._batch_first = True if batch_first else False

        self._weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self._weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(hidden_size))
        self._bias_hh = Parameter(torch.randn(hidden_size))
        self.reset_parameters()

        self.register_buffer(
            "_mask_i2h", torch.ones((hidden_size, self._input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((hidden_size, self._hidden_size), dtype=torch.bool)
        )

    def forward(self, input, hx):
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        return self._nonlinearity(igate + hgate)

    def extra_repr(self):
        s = "in_features={_input_size}, out_features={_hidden_size}, nonlinearity={_nonlinearity}"

        if self._batch_first:
            s += ", batch_first={_batch_first}"

        return s.format(**self.__dict__)


class MaskedDeepRNN(BaseMaskModule):
    """
    A deep Vanilla RNN model with maskable layers to allow for sparsity.

    Args:
        input_size: The number of expected features in the input
        hidden_layers: A list specifying number of expected features in each hidden layer
            (E.g, hidden_layers=[50, 50] specifies model consisting two hidden layers with 50 features each)
        nonlinearity: Can be either 'tanh', 'relu'
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self,
        input_size,
        hidden_layers: list,
        nonlinearity=torch.nn.Tanh(),
        batch_first=False,
    ):
        super(MaskedDeepRNN, self).__init__()

        assert callable(nonlinearity)

        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._nonlinearity = nonlinearity
        self._batch_first = batch_first

        self._recurrent_layers = nn.ModuleList()
        for lay, hidden_size in enumerate(hidden_layers):
            input_size = input_size if lay == 0 else hidden_layers[lay - 1]
            self._recurrent_layers.append(
                MaskedRecurrentLayer(input_size, hidden_size, nonlinearity, batch_first)
            )

    def forward(self, input):
        batch_size = input.size(0) if self._batch_first else input.size(1)

        for lay, hidden_size in enumerate(self._hidden_layers):
            hx = torch.zeros(
                batch_size, hidden_size, dtype=input.dtype, device=input.device
            )
            input = self.__step(self._recurrent_layers[lay], input, hx)

        output = input[:, -1, :] if self._batch_first else input[-1]
        return output

    def __step(self, layer, input, hx):
        in_dim = 1 if self._batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self._batch_first else input[i]
            hx = layer(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class MaskedGRULayer(BaseLayerModule):
    """
    Base class for layer initialization for GRU.

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
    ):
        super(MaskedGRULayer, self).__init__()

        assert input_size > 0
        assert hidden_size > 0

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._batch_first = True if batch_first else False
        gate_size = 3 * hidden_size

        self._weight_ih = Parameter(torch.randn(gate_size, input_size))
        self._weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(gate_size))
        self._bias_hh = Parameter(torch.randn(gate_size))
        self.reset_parameters()

        self.register_buffer(
            "_mask_i2h", torch.ones((gate_size, self._input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((gate_size, self._hidden_size), dtype=torch.bool)
        )

    def forward(self, input, hx):
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        i_reset, i_input, i_new = igate.chunk(3, 1)
        h_reset, h_input, h_new = hgate.chunk(3, 1)

        reset_gate = torch.sigmoid(i_reset + h_reset)
        input_gate = torch.sigmoid(i_input + h_input)
        new_gate = torch.tanh(i_new + reset_gate * h_new)

        hx = new_gate + input_gate * (hx - new_gate)
        return hx

    def extra_repr(self):
        s = "in_features={_input_size}, out_features={_hidden_size}"

        if self._batch_first:
            s += ", batch_first={_batch_first}"

        return s.format(**self.__dict__)


class MaskedDeepGRU(BaseMaskModule):
    """
    A deep GRU model with maskable layers to allow for sparsity.

    Args:
        input_size: The number of expected features in the input
        hidden_layers: A list specifying number of expected features in each hidden layer
            (E.g, hidden_layers=[50, 50] specifies model consisting two hidden layers with 50 features each)
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(self, input_size, hidden_layers: list, batch_first=False):
        super(MaskedDeepGRU, self).__init__()

        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._batch_first = batch_first

        self._recurrent_layers = nn.ModuleList()
        for lay, hidden_size in enumerate(hidden_layers):
            input_size = input_size if lay == 0 else hidden_layers[lay - 1]
            self._recurrent_layers.append(
                MaskedGRULayer(input_size, hidden_size, batch_first)
            )

    def forward(self, input):
        batch_size = input.size(0) if self._batch_first else input.size(1)

        for lay, hidden_size in enumerate(self._hidden_layers):
            hx = torch.zeros(
                batch_size, hidden_size, dtype=input.dtype, device=input.device
            )
            input = self.__step(self._recurrent_layers[lay], input, hx)

        output = input[:, -1, :] if self._batch_first else input[-1]
        return output

    def __step(self, layer, input, hx):
        in_dim = 1 if self._batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self._batch_first else input[i]
            hx = layer(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class MaskedLSTMLayer(BaseLayerModule):
    """
    Base class for layer initialization for LSTM.

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
    ):
        super(MaskedLSTMLayer, self).__init__()

        assert input_size > 0
        assert hidden_size > 0

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._batch_first = True if batch_first else False
        gate_size = 4 * hidden_size

        self._weight_ih = Parameter(torch.randn(gate_size, input_size))
        self._weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(gate_size))
        self._bias_hh = Parameter(torch.randn(gate_size))
        self.reset_parameters()

        self.register_buffer(
            "_mask_i2h", torch.ones((gate_size, self._input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((gate_size, self._hidden_size), dtype=torch.bool)
        )

    def forward(self, input, hx):
        hx, cx = hx
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        gates = igate + hgate

        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cx = (forget_gate * cx) + (input_gate * cell_gate)
        hx = out_gate * torch.tanh(cx)
        return hx, cx

    def extra_repr(self):
        s = "in_features={_input_size}, out_features={_hidden_size}"

        if self._batch_first:
            s += ", batch_first={_batch_first}"

        return s.format(**self.__dict__)


class MaskedDeepLSTM(BaseMaskModule):
    """
    A deep LSTM model with maskable layers to allow for sparsity.

    Args:
        input_size: The number of expected features in the input
        hidden_layers: A list specifying number of expected features in each hidden layer
            (E.g, hidden_layers=[50, 50] specifies model consisting two hidden layers with 50 features each)
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(self, input_size, hidden_layers: list, batch_first=False):
        super(MaskedDeepLSTM, self).__init__()

        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._batch_first = batch_first

        self._recurrent_layers = nn.ModuleList()
        for lay, hidden_size in enumerate(hidden_layers):
            input_size = input_size if lay == 0 else hidden_layers[lay - 1]
            self._recurrent_layers.append(
                MaskedLSTMLayer(input_size, hidden_size, batch_first)
            )

    def forward(self, input):
        batch_size = input.size(0) if self._batch_first else input.size(1)

        for lay, hidden_size in enumerate(self._hidden_layers):
            hx = torch.zeros(
                batch_size, hidden_size, dtype=input.dtype, device=input.device
            )
            input = self.__step(self._recurrent_layers[lay], input, hx)

        output = input[:, -1, :] if self._batch_first else input[-1]
        return output

    def __step(self, layer, input, hx):
        in_dim = 1 if self._batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []
        cx = hx.clone()

        for i in range(n_seq):
            seq = input[:, i, :] if self._batch_first else input[i]
            hx, cx = layer(seq, (hx, cx))
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)
