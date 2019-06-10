import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from model.vae import VAE


class Inference(object):
    def __init__(self, binary_features, continuous_features, z_dim, hidden_dim, hidden_layers, learning_rate, activation, cuda):
        pyro.clear_param_store()
        vae = VAE(binary_features, continuous_features, z_dim,
                  hidden_dim, hidden_layers, activation, cuda)
        optimizer = Adam({
            "lr": learning_rate
        })
        self.svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
        self.cuda = cuda

    def train(self, train_loader):
        epoch_loss = 0.
        for mu1, mu0, t, x, yf, ycf in train_loader:
            if self.cuda:
                x = x.cuda()
            epoch_loss += self.svi.step((x, t, yf))

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train

        return total_epoch_loss_train

    def evaluate(self, test_loader):
        test_loss = 0.
        for mu1, mu0, t, x, yf, ycf in test_loader:
            if self.cuda:
                x = x.cuda()
            test_loss += self.svi.evaluate_loss((x, t, yf))

        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test

        return total_epoch_loss_test
