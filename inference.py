import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from model.vae import VAE
from statistics import Statistics


class Inference(object):
    def __init__(self,
                 y_mean, y_std, binary_features, continuous_features,
                 z_dim, hidden_dim, hidden_layers, optimizer, activation,
                 cuda):
        pyro.clear_param_store()
        vae = VAE(binary_features, continuous_features, z_dim,
                  hidden_dim, hidden_layers, activation, cuda)
        vae = vae.double()
        self.vae = vae
        self.svi = SVI(vae.model, vae.guide,
                       optimizer, loss=Trace_ELBO())
        self.cuda = cuda
        self.y_mean = y_mean
        self.y_std = y_std

        self.train_stats = Statistics()
        self.test_stats = Statistics()

    def train(self, train_loader):
        epoch_loss = 0.
        for mu1, mu0, t, x, yf, ycf, std_yf in train_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda(), std_yf.cuda()
            epoch_loss += self.svi.step((x, t, std_yf))
            self.train_stats.collect('mu1', mu1)
            self.train_stats.collect('mu0', mu0)
            self.train_stats.collect('t', t)
            self.train_stats.collect('x', x)
            self.train_stats.collect('yf', yf)
            self.train_stats.collect('ycf', ycf)

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train

        return total_epoch_loss_train

    def evaluate(self, test_loader):
        test_loss = 0.
        for mu1, mu0, t, x, yf, ycf, std_yf in test_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda(), std_yf.cuda()
            test_loss += self.svi.evaluate_loss((x, t, std_yf))
            self.test_stats.collect('mu1', mu1)
            self.test_stats.collect('mu0', mu0)
            self.test_stats.collect('t', t)
            self.test_stats.collect('x', x)
            self.test_stats.collect('yf', yf)
            self.test_stats.collect('ycf', ycf)

        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test

        return total_epoch_loss_test

    def _predict(self, x, L):
        y0, y1 = self.vae.predict_y(x, L)

        return self.y_mean + y0 * self.y_std, self.y_mean + y1 * self.y_std

    def train_statistics(self, L):
        y0, y1 = self._predict(self.train_stats.data['x'], L)
        ITE, ATE, PEHE = self.train_stats.calculate(y0, y1)
        RMSE_factual, RMSE_counterfactual = self.train_stats.y_errors(y0, y1)

        return (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual)

    def test_statistics(self, L):
        y0, y1 = self._predict(self.test_stats.data['x'], L)
        ITE, ATE, PEHE = self.test_stats.calculate(y0, y1)
        RMSE_factual, RMSE_counterfactual = self.test_stats.y_errors(y0, y1)

        return (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual)
