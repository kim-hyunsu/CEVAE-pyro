import pyro
from pyro.infer import SVI, Trace_ELBO
from model.vae import VAE


class Inference(object):
    def __init__(self, y_mean, y_std, binary_features, continuous_features, z_dim, hidden_dim, hidden_layers, optimizer, activation, cuda):
        pyro.clear_param_store()
        vae = VAE(binary_features, continuous_features, z_dim,
                  hidden_dim, hidden_layers, activation, cuda)
        vae = vae.double()
        self.svi = SVI(vae.model, vae.guide,
                       optimizer, loss=Trace_ELBO())
        self.cuda = cuda
        self.y_mean = y_mean
        self.y_std = y_std

    def train(self, train_loader):
        epoch_loss = 0.
        for mu1, mu0, t, x, yf, ycf in train_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda()
            epoch_loss += self.svi.step((x, t, yf))

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train

        return total_epoch_loss_train

    def evaluate(self, test_loader):
        test_loss = 0.
        for mu1, mu0, t, x, yf, ycf in test_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda()
            test_loss += self.svi.evaluate_loss((x, t, yf))

            y0, y1 = self.svi.predict_y(x, L=1)
            y0, y1 = self.y_mean + y0 * self.y_std, self.y_mean + y1 * self.y_std

            # TODO: evaluators

        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test

        return total_epoch_loss_test

    def _RMSE_ITE(self, y0, y1):
        pass

    def _ABS_ATE(self, y0, y1):
        pass

    def _PEHE(self, y0, y1):
        pass

    def _y_errors(self, y0, y1):
        pass

    def _y_errors(self, y_factual, y_counterfactual):
        pass
