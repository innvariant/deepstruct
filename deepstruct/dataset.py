import pickle

import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class FuncDataset(Dataset):
    def __init__(self, fn, size: int, shape_input: tuple = None, sampler=None):
        assert fn is not None
        assert callable(fn)
        assert shape_input is not None
        assert size is not None and size > 0
        assert callable(sampler) or sampler is None

        self._size = size
        self._sampler = sampler
        self._fn = fn
        self._shape_input = shape_input
        self._data = None

    def generate(self, without_progress: bool = True):
        self._data = []
        has_length = hasattr(self._fn, "__len__")
        fn_name = str(self._fn)
        with tqdm(
            total=self._size, desc="Generating samples", disable=without_progress
        ) as pbar:
            for ix in range(self._size):
                self._data.append(self.sample())
                if has_length:
                    pbar.set_postfix(**{fn_name: len(self._fn)})
                pbar.update()

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        assert sampler is not None and callable(sampler)
        self._sampler = sampler

    def sample(self):
        assert self._sampler is not None
        preimage_sample = torch.tensor(self._sampler()).view(self._shape_input)
        # print(preimage_sample.shape)
        codomain = self._fn(preimage_sample)
        return preimage_sample, codomain

    def save(self, path_file):
        with open(path_file, "wb") as handle:
            pickle.dump(
                {
                    "size": self._size,
                    "shape_input": self._shape_input,
                    "data": self._data,
                },
                handle,
            )

    @staticmethod
    def load(path_file):
        with open(path_file, "rb") as handle:
            meta = pickle.load(handle)
        dataset = FuncDataset(
            lambda: None, meta["size"], meta["shape_input"], lambda: None
        )
        dataset._data = meta["data"]
        return dataset

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int):
        if self._data is None:
            self.generate()
        assert self._data is not None

        idx = idx + int(idx < 0) * len(self)

        if idx > len(self):
            raise StopIteration("Given index <%s> exceeds dataset." % idx)

        return self._data[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from deepstruct.sparse import MaskedDeepFFN

    batch_size = 100
    ds_input_shape = (2,)
    # fn_target = lambda x: np.array([4+x[0]**2-3*x[1]])
    # fn_target = lambda x: np.array([4 + x[0] ** 2 - 3 * x[1]])
    stier2020B1d = (
        lambda x, y: 20
        + x
        - 1.8 * (y - 5)
        + 3 * np.sin(x + 2 * y) * y
        + (x / 4) ** 4
        + (y / 4) ** 4
    )

    def fn_target(x):
        return np.array([stier2020B1d(x[0], x[1])])

    # Training
    ds_train = FuncDataset(fn_target, shape_input=ds_input_shape, size=500)
    ds_train.sampler = lambda: np.random.uniform(-2, 2, size=ds_input_shape)

    ds_output_shape = fn_target(ds_train.sampler()).shape
    print("f: R^(%s) --> R^(%s)" % (ds_input_shape, ds_output_shape))

    train_sampler = torch.utils.data.SubsetRandomSampler(
        np.arange(len(ds_train), dtype=np.int64)
    )
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )

    model = MaskedDeepFFN(ds_input_shape, 1, [100, 100])

    fn_loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.train()
    errors_train = []
    for epoch in range(100):
        for feat, target in train_loader:
            target = target.reshape(-1, 1)
            optimizer.zero_grad()
            pred = model(feat)
            error = fn_loss(target, pred)
            errors_train.append(error.detach().numpy())
            error.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Cur Err [-1]:", errors_train[-1])
            print("Avg Err [-5]:", np.mean(errors_train[-5:]))

    # Testing
    ds_test = FuncDataset(fn_target, shape_input=ds_input_shape, size=5000)
    ds_test.sampler = lambda: np.random.uniform(-3, 3, size=ds_input_shape)

    test_sampler = torch.utils.data.SubsetRandomSampler(
        np.arange(len(ds_test), dtype=np.int64)
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, sampler=test_sampler, num_workers=2
    )

    model.eval()
    errors_test = []
    xs = np.array([]).reshape((-1,) + ds_input_shape)
    ys = np.array([]).reshape((-1,) + ds_output_shape)
    ms = np.array([]).reshape((-1,) + ds_output_shape)
    for feat, target in test_loader:
        target = target.reshape(-1, 1)
        pred = model(feat)
        error = fn_loss(target, pred)
        errors_test.append(error.detach().numpy())

        xs = np.vstack([xs, feat.detach().numpy()])
        ys = np.vstack([ys, target.detach().numpy()])
        ms = np.vstack([ms, pred.detach().numpy()])

    print(errors_test)
    print(np.mean(errors_test))

    # print(xs.shape)
    # print(ys.shape)
    # print(ms.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="blue")
    ax.scatter(xs[:, 0], xs[:, 1], ms, marker=".", color="orange")
    # plt.plot(xs, ys)
    # plt.plot(xs, ms)
    ax.set_zlim(0, 50)
    plt.show()
