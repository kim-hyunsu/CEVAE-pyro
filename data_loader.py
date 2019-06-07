import numpy as np
import torch.utils.data


class IHDPDataset(torch.utils.data.Dataset):
    def __init__(self, np_data):
        self.mu1 = np_data['mu1']
        self.mu0 = np_data['mu0']
        self.t = np_data['t']
        self.x = np_data['x']
        self.yf = np_data['yf']
        self.ycf = np_data['ycf']
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.mu1[idx], self.mu0[idx], self.t[idx], self.x[idx], self.yf[idx], self.ycf[idx]

    def __len__(self):
        return self.len


class IHDPLoader(object):
    def __init__(self, path="data/IHDP-np", cuda=False):
        self.path = path
        self.cuda = cuda

        test_data = np.load(f'{path}/ihdp_npci_1-100.test.npz')
        train_data = np.load(f'{path}/ihdp_npci_1-100.train.npz')

        self._indices_binary_continuous(train_data['x'][1, :])
        self.test_set = IHDPDataset(test_data)
        self.train_set = IHDPDataset(train_data)

    def _indices_binary_continuous(self, features):
        binary_indices = []
        continuous_indices = []
        for i, x in enumerate(features):
            binary = True
            for num in x:
                if not num in (0, 1, 2):
                    binary = False
            if binary:
                binary_indices.append(i)
            else:
                continuous_indices.append(i)

        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices

    def indices_each_features(self):
        return self.binary_indices, self.continuous_indices

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
