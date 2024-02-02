import torch.nn

import deepstruct.pruning.engine as dpeng
import deepstruct.sparse as dsp


def test_engine_construct():
    model = dsp.MaskedDeepFFN(784, 10, [100], use_layer_norm=True)
    criterion = torch.nn.CrossEntropyLoss()

    def prune_step(engine, batch):
        model.eval()
        inputs, targets = batch[0].cuda(), batch[1].cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        model.saliency = loss

    dpeng.Engine(prune_step)
