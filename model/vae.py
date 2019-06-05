import models


class VAE(nn.Module):
    def __init__(self, z_dim, insize, d, nh, h, binfeats, contfeats, activation, cuda=False):
        super(VAE, self).__init__()
        self.encoder = models.Encoder(
            in_size, in_size+1, d, nh, h, binfeats, contfeats, activation)
        self.decoder = models.Decoder(
            d, nh, h, binfeats, contfeats, activation)

        if cuda:
            self.cuda()
        self.cuda = cuda
        self.z_dim = z_dim
        self.binfeats = binfeats
        self.contfeats = contfeats

    def model(self, data):
        pyro.module("decoder", self.decoder)
        binary_x_observation = data[:, :self.binfeats]
        continuous_x_observation = data[:,
                                        self.binfeats:self.binfeats+self.contfeats]
        t_observation = data[:, -2]
        y_observation = data[:, -1]
        with pyro.plate("data", data.shape[0]):
            z_loc = data.new_zeros(torch.Size((data.shape[0], self.z_dim)))
            z_scale = data.new_ones(torch.Size((data.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # P(x|z) for binary x
            x_logists = self.decoder.forward_P_x(z)
            pyro.sample('x_bin', dist.Bernoulli(
                logists_1).to_event(1), obs=binary_x_observation)
            # P(x|z) for continuous x
            x_loc, x_scale = self.decoder.forward_P_x(z, cont=True)
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(
                1), obs=continuous_x_observation)
            # P(t|z)
            t_logits = self.decoder.forward_P_t(z)
            t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1),
                            obs=t_observation.contiguous().view(-1, 1))
            # P(y|z, t)
            y_loc, y_scale = self.decoder.forward_P_y(z, t)
            if self.cuda:
                y_scale = y_scale.cuda()
            pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1),
                        obs=y_observation.contiguous().view(-1, 1))

    def guide(self, data):
        pyro.module("encoder", self.encoder)
        x_observation = data[:, :self.binfeats + self.contfeats]
        with pyro.plate("data", data.shape[0]):
            # Q(t|x)
            logits_t = self.encoder.forward_Q_t(x_observation):
            t = sample('t', dist.Bernoulli(logits_t).to_event(1))
            # Q(y|x,t)
            y_loc, y_scale = self.encoder.forward_Q_y(x_observation, t)
            if self.cuda:
                y_scale = y_scale.cuda()
            y = sample('y', dist.Normal(y_loc, y_scale))
            # Q(z|x,t,y)
            z_loc, z_scale = self.encoder.forward_Q_z(x_observation, y, t)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
