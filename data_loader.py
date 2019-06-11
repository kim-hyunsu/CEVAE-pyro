import numpy as np
import torch.utils.data


class IHDPNPZDataset(torch.utils.data.Dataset):
    def __init__(self, np_data):
        self.mu1 = np_data['mu1'][:, 1]
        self.mu0 = np_data['mu0'][:, 1]
        self.t = np_data['t'][:, 1]
        self.x = np_data['x'][:, 1]
        self.yf = np_data['yf'][:, 1]
        self.ycf = np_data['ycf'][:, 1]
        self.len = self.x.shape[0]
        self.binary_indices = []
        self.continuous_indices = []
        for i, x in enumerate(self.x[1, :]):
            if x in (0, 1, 2):
                self.binary_indices.append(i)
            else:
                self.continuous_indices.append(i)

    def __getitem__(self, idx):
        return self.mu1[idx], self.mu0[idx], self.t[idx], self.x[idx], self.yf[idx], self.ycf[idx]

    def __len__(self):
        return self.len

    def indices_each_features(self):
        return self.binary_indices, self.continuous_indices


class DataLoader(object):
    def __init__(self, train, test, cuda=False):
        self.cuda = cuda
        self.test_set = test
        self.train_set = train

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set, batch_size=batch_size)

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=self.cuda)

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader
