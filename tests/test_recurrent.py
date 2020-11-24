import numpy as np
import torch

import deepstruct.recurrent
import deepstruct.sparse


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


def test_recurrent_unusual_activation():
    # Arrange
    batch_size = 15
    input_size = 20
    hidden_size = 30
    model = deepstruct.recurrent.MaskedRecurrentLayer(
        input_size, hidden_size=hidden_size, nonlinearity=torch.nn.LogSigmoid()
    )
    random_input = torch.tensor(
        np.random.random((batch_size, input_size)),
        dtype=torch.float32,
        requires_grad=False,
    )

    # Act
    hidden_state = torch.tensor(
        np.random.random((batch_size, hidden_size)),
        dtype=torch.float32,
        requires_grad=False,
    )
    output = model(random_input, hidden_state)

    # Assert
    assert output.numel() == batch_size * hidden_size


def test_deep_recurrent_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    batch_size = 15
    input_size = 20
    seq_size = 18
    output_size = 12
    model = deepstruct.recurrent.MaskedDeepRNN(
        input_size, hidden_layers=[100, 100, output_size], batch_first=True
    )
    model.to(device)
    random_input = torch.tensor(
        np.random.random((batch_size, seq_size, input_size)),
        # np.random.random((seq_size, batch_size, input_size)),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    print(str(model))

    # Act
    result_shape = model(random_input).shape

    assert result_shape.numel() == batch_size * output_size


def test_deep_recurrent_layertypes_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    batch_size = 15
    input_size = 20
    seq_size = 18
    output_size = 12
    layer_constructors = [
        deepstruct.recurrent.MaskedRecurrentLayer,
        deepstruct.recurrent.MaskedGRULayer,
        deepstruct.recurrent.MaskedLSTMLayer,
    ]
    for builder in layer_constructors:
        model = deepstruct.recurrent.MaskedDeepRNN(
            input_size,
            hidden_layers=[100, 100, output_size],
            batch_first=True,
            build_recurrent_layer=builder,
        )
        model.to(device)
        random_input = torch.tensor(
            np.random.random((batch_size, seq_size, input_size)),
            # np.random.random((seq_size, batch_size, input_size)),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )

        # Act
        result_shape = model(random_input).shape

        assert result_shape.numel() == batch_size * output_size


def test_learn_start_symbol():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    n_samples = 200
    batch_size = 15
    input_size = 1
    seq_size = 20
    output_size = 10

    # Build up samples like
    #  [1, 2, 3, 4, 5, 0, 0, 0] --> 1
    #  [3, 4, 5, 6, 0, 0, 0, 0] --> 3
    samples_x = []
    samples_y = []
    for sample_idx in range(n_samples):
        start = np.random.randint(1, 20) + 1
        length = np.random.randint(2, seq_size + 1)
        x_seq = np.pad(
            np.arange(start, start + length), (0, seq_size - length), "constant"
        ).reshape(seq_size, 1)
        y_result = start
        samples_x.append(x_seq)
        samples_y.append(y_result)

    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)

    # Define model, loss and optimizer
    model = deepstruct.recurrent.MaskedDeepRNN(
        input_size,
        hidden_layers=[30, output_size],
        batch_first=True,
        build_recurrent_layer=deepstruct.recurrent.MaskedLSTMLayer,
    )
    model = torch.nn.Sequential(model, torch.nn.Linear(output_size, 1))
    model.to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Perform training with backpropagation
    for epoch in range(100):
        errors = []
        for batch_idx in range(int(n_samples / batch_size)):
            optimizer.zero_grad()

            output = model(
                torch.tensor(
                    samples_x[batch_idx * 15 : 15 + batch_idx * 15],
                    dtype=torch.float32,
                    device=device,
                )
            )
            expected = torch.tensor(
                samples_y[batch_idx * 15 : 15 + batch_idx * 15].reshape(batch_size, 1),
                dtype=torch.float32,
                device=device,
            )

            error = loss(output, expected)
            error.backward()
            errors.append(error.detach().cpu().numpy())

            optimizer.step()

    for start, length in [(1, 3), (5, 10), (8, 5)]:
        x_seq = np.pad(
            np.arange(start, start + length), (0, seq_size - length), "constant"
        ).reshape(1, seq_size, 1)
        prediction = model(torch.tensor(x_seq, dtype=torch.float32, device=device))
        target = start
        print("Test: target=", target, "prediction=", prediction)


def test_learn_summation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    n_samples = 200
    batch_size = 15
    input_size = 1
    seq_size = 20
    output_size = 10

    # Build up samples like
    #  [1, 2, 3, 4, 5, 0, 0, 0] --> 15
    #  [3, 4, 5, 6, 0, 0, 0, 0] --> 18
    def gauss(n):
        return (n * (n + 1)) / 2

    samples_x = []
    samples_y = []
    for sample_idx in range(n_samples):
        start = np.random.randint(1, 20)
        length = np.random.randint(2, seq_size + 1)
        x_seq = np.pad(
            np.arange(start, start + length), (0, seq_size - length), "constant"
        ).reshape(seq_size, 1)
        y_result = gauss(start + length - 1) - gauss(start - 1)
        samples_x.append(x_seq)
        samples_y.append(y_result)

    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)

    # Define model, loss and optimizer
    model = deepstruct.recurrent.MaskedDeepRNN(
        input_size,
        hidden_layers=[50, output_size],
        batch_first=True,
        nonlinearity=torch.nn.ReLU(),
    )
    model = torch.nn.Sequential(model, torch.nn.Linear(output_size, 1))
    model.to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Perform training with backpropagation
    for epoch in range(100):
        errors = []
        for batch_idx in range(int(n_samples / batch_size)):
            optimizer.zero_grad()

            output = model(
                torch.tensor(
                    samples_x[batch_idx * 15 : 15 + batch_idx * 15],
                    dtype=torch.float32,
                    device=device,
                )
            )
            expected = torch.tensor(
                samples_y[batch_idx * 15 : 15 + batch_idx * 15].reshape(batch_size, 1),
                dtype=torch.float32,
                device=device,
            )

            error = loss(output, expected)
            error.backward()
            errors.append(error.detach().cpu().numpy())

            optimizer.step()

    for start, length in [(1, 5), (5, 7), (8, 4)]:
        x_seq = np.pad(
            np.arange(start, start + length), (0, seq_size - length), "constant"
        ).reshape(1, seq_size, 1)
        prediction = model(torch.tensor(x_seq, dtype=torch.float32, device=device))
        target = gauss(start + length - 1) - gauss(start - 1)
        print("Test: target=", target, "prediction=", prediction)
