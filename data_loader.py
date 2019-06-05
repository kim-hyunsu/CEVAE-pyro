import numpy as np
import torch


class IHDPLoader(object):
    def __init__(self, path="data/IHDP", cuda=False):
        self.path = path
        self.binary_features = list(range(6, 25))
        self.continuous_features = list(range(7))

        for i in range(10):
            data = np.loadtxt(
                f'{self.path}/ihdp_npci_{i % 10 + 1}.csv', delimiter=',')
            # TODO

    def num_each_features(self):
        return self.binary_features, self.continuous_features

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set, batch_size=batch_size)

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set, batch_size=batch_size, shuffle=False, {
                'num_workers': 1, 'pin_memory': self.cuda
            })

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader
