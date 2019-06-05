class DataLoader(object):
    def __init__(self, name, path, cuda):
        self.path = path

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader()

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader()

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader
