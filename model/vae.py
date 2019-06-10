from torch import nn
import networks


class VAE(nn.Module):
    def __init__(self, binary_features, continuous_features, z_dim, hidden_dim, hidden_layers, activation, cuda):
        super(VAE, self).__init__()
        self.encoder = networks.Encoder(
            binary_features, continuous_features, z_dim, hidden_dim, hidden_layers, activation, cuda)
        self.decoder = networks.Decoder(
            binary_features, continuous_features, z_dim, hidden_dim, hidden_layers, activation, cuda)

        if cuda:
            self.cuda()
        self.cuda = cuda
        self.z_dim = z_dim
        self.binary = binary_features
        self.continuous = continuous_features

    def model(self, data):
        pyro.module("decoder", self.decoder)
        x_observation = data[0]
        binary_x_observation = data[:, :self.binary]
        continuous_x_observation = data[:,
                                        self.binary:self.binary+self.continuous]
        t_observation = data[1]
        y_observation = data[2]
        with pyro.plate("data", x_observation.shape[0]):
            z_loc = data.new_zeros(torch.Size(
                (x_observation.shape[0], self.z_dim)))
            z_scale = data.new_ones(torch.Size(
                (x_observation.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0,
                                                     y_loc_t1, y_scale) = self.decoder.forward(z)
            # P(x|z) for binary x
            pyro.sample('x_bin', dist.Bernoulli(
                x_logits).to_event(1), obs=binary_x_observation)

            # P(x|z) for continuous x
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(
                1), obs=continuous_x_observation)

            # P(t|z)
            t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1),
                            obs=t_observation.contiguous().view(-1, 1))

            # P(y|z, t)
            y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
            pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1),
                        obs=y_observation.contiguous().view(-1, 1))

    def guide(self, data):
        pyro.module("encoder", self.encoder)
        x_observation = data[0][:, :self.binary + self.continuous]
        with pyro.plate("data", data.shape[0]):
            z_loc, z_scale = self.encoder.forward(x_observation)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
