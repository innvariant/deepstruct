import numpy as np
import torch

import deepstruct.recurrent


def test_recurrent_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    batch_size = 15
    input_size = 20
    hidden_size = 30
    model = deepstruct.recurrent.MaskedRecurrentLayer(
        input_size, hidden_size=hidden_size
    )
    model.to(device)
    random_input = torch.tensor(
        np.random.random((batch_size, input_size)),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    # Act
    hidden_state = torch.tensor(
        np.random.random((batch_size, hidden_size)),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    output = model(random_input, hidden_state)

    # Assert
    assert output.numel() == batch_size * hidden_size
