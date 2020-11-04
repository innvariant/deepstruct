import torch


class TestTensorsData:

    # Input Tensor's here.

    @staticmethod
    def get_max_pool_input_tensor1():
        temp_tensor = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
                    [0.4, 0.5, 0.2, 0.3, 0.7, 0.5],
                    [0.1, 0.7, 0.6, 0.2, 0.9, 0.1],
                    [0.3, 0.3, 0.1, 0.6, 0.5, 0.7],
                    [0.0, 0.6, 0.2, 0.8, 0.1, 0.2],
                    [0.0, 0.6, 0.2, 0.8, 0.1, 0.2],
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_max_pool_input_tensor2():
        temp_tensor = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
                    [0.4, 0.5, 0.2, 0.3, 0.7, 0.5],
                    [0.1, 0.7, 0.6, 0.2, 0.9, 0.1],
                    [0.3, 0.3, 0.0, 0.0, 0.5, 0.7],
                    [0.0, 0.6, 0.0, 0.0, 0.1, 0.2],
                    [0.0, 0.6, 0.0, 0.0, 0.1, 0.2],
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_conv_max_pool_input_tensor():
        temp_tensor = torch.tensor(
            [
                [
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.2, 0.3, 0.7],
                        [0.1, 0.7, 0.6, 0.2, 0.9],
                        [0.3, 0.3, 0.1, 0.6, 0.5],
                        [0.0, 0.6, 0.2, 0.8, 0.1],
                    ]
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_conv_input_tensor():
        temp_tensor = torch.tensor(
            [
                [
                    [
                        [0.1, 0.2, 0.3, 0.8, 0.4],
                        [0.0, 0.0, 0.2, 0.8, 0.4],
                        [0.2, 0.9, 0.3, 0.0, 0.5],
                        [0.0, 0.6, 0.6, 0.9, 0.4],
                        [0.1, 0.0, 0.6, 0.8, 0.4],
                    ]
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_conv_input_tensor_2_1():
        temp_tensor = torch.tensor(
            [
                [
                    [
                        [0.1, 0.2, 0.3, 0.8, 0.4],
                        [0.0, 0.0, 0.2, 0.8, 0.4],
                        [0.2, 0.9, 0.3, 0.0, 0.5],
                        [0.0, 0.6, 0.6, 0.9, 0.4],
                        [0.1, 0.0, 0.6, 0.8, 0.4],
                    ],
                    [
                        [0.1, 0.2, 0.3, 0.8, 0.4],
                        [0.0, 0.0, 0.2, 0.8, 0.4],
                        [0.2, 0.9, 0.3, 0.0, 0.5],
                        [0.0, 0.6, 0.6, 0.9, 0.4],
                        [0.1, 0.0, 0.6, 0.8, 0.4],
                    ],
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_Linear_input_tensor():
        temp_tensor = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.2],
            ]
        )
        return temp_tensor

    # Weights Here

    @staticmethod
    def get_conv_weights_1_3():
        temp_tensor = torch.tensor(
            [
                [[[0.1, 0.6], [0.1, 0.0]]],
                [[[0.1, 0.2], [0.0, 0.9]]],
                [[[0.0, 0.2], [0.1, 0.7]]],
            ]
        )
        return temp_tensor

    @staticmethod
    def get_conv_weights_1_2():
        temp_tensor = torch.tensor(
            [
                [
                    [
                        [0.1, 0.2, 0.3],
                        [0.1, 0.1, 0.2],
                        [0.1, 0.1, 0.6],
                    ]
                ],
                [
                    [
                        [0.1, 0.2, 0.3],
                        [0.1, 0.1, 0.2],
                        [0.1, 0.1, 0.6],
                    ]
                ],
            ]
        )
        return temp_tensor

    @staticmethod
    def get_conv_weights_2_1():
        temp_tensor = torch.tensor(
            [
                [
                    [[0.0, 0.4, 0.7], [0.1, 0.2, 0.3], [0.9, 0.2, 0.3]],
                    [[0.4, 0.6, 0.5], [0.2, 0.1, 0.1], [0.1, 0.1, 0.1]],
                ]
            ]
        )
        return temp_tensor

    @staticmethod
    def get_input_tensor_weight_6_3():
        temp_tesnsor = torch.tensor(
            [
                [0.1, 0.8, 0.0, 0.0, 0.9, 0.0],
                [0.1, 0.2, 0.2, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.1, 0.0, 0.0, 0.0],
            ]
        )
        return temp_tesnsor

    # Bias

    @staticmethod
    def get_bias_1():
        temp_tensor = torch.tensor([0.0])
        return temp_tensor

    @staticmethod
    def get_bias_2():
        temp_tensor = torch.tensor([0.0, 0.0])
        return temp_tensor

    @staticmethod
    def get_bias_3():
        temp_tensor = torch.tensor([0.0, 0.0, 0.0])
        return temp_tensor
