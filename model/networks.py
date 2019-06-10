import torch
import torch.nn.functional as F
from torch import nn


class FCNet(nn.Module):
    def __init__(self, input_size, layers, out_layers, activation):
        super(FCNet, self).__init__()

        self.activation = activation

        if layers:
            self.input = nn.Linear(input_size, layers[0])
            self.hidden_layers = nn.ModuleList()
            for i in range(1, len(layers)):
                self.hidden_layers.append(nn.Linear(layers[i], layers[i]))
            self.output_layers = nn.ModuleList()
            self.output_activations = []
            for i, (outdim, self.activation) in enumerate(out_layers):
                self.output_layers.append(nn.Linear(layers[-1], outdim))
                self.output_activations.append(self.activation)
        else:
            self.output_layers = nn.ModuleList()
            self.output_activations = []
            for i, (outdim, self.activation) in enumerate(out_layers):
                self.output_layers.append(nn.Linear(input_size, outdim))
                self.output_activations.append(self.activation)

    def forward(self, x):

        x = self.activation(self.input(x))
        try:
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
        except AttributeError:
            pass
        if self.output_layers:
            outputs = []
            for output_layer, output_activation in zip(self.output_layers, self.output_activations):
                if output_activation:
                    outputs.append(output_activation(output_layer(x)))
                else:
                    outputs.append(output_layer(x))
            return outputs if len(outputs) > 1 else outputs[0]
        else:
            return x


class Decoder(nn.Module):
    def __init__(self, binfeats, contfeats, n_z, h, nh, activation, cuda):
        super(Decoder, self).__init__()
        self.cuda = cuda

        # p(x|z)
        self.hx = FCNet(n_z, (nh - 1) * [h], [], activation)
        self.logits_1 = FCNet(h, [h], [[binfeats, F.sigmoid]], activation)

        self.mu_sigma = FCNet(
            h, [h], [[contfeats, None], [contfeats, F.softplus]], activation)

        # p(t|z)
        self.logits_2 = FCNet(n_z, [h], [[1, F.sigmoid]], activation)

        # p(y|t,z)
        self.mu2_t0 = FCNet(n_z, nh * [h], [[1, None]], activation)
        self.mu2_t1 = FCNet(n_z, nh * [h], [[1, None]], activation)

    def forward(self, z):
        # p(x|z)
        hx = self.hx.forward(z)
        x_logits = self.logits_1.forward(hx)
        x_loc, x_scale = self.mu_sigma.forward(hx)

        # p(t|z)
        t_logits = self.logits_2(z)

        # p(y|t,z)
        y_loc_t0 = self.mu2_t0(z)
        y_loc_t1 = self.mu2_t1(z)
        y_scale = Variable(torch.ones(mu2_t0.size()))
        if self.cuda:
            y_scale = y_scale.cuda()

        return (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0, y_loc_t1, y_scale)

    def P_y_zt(self, z, t):
        # p(y|t,z)
        mu2_t0 = self.mu2_t0(z)
        mu2_t1 = self.mu2_t1(z)

        sig = Variable(torch.ones(mu2_t0.size()))
        if mu2_t0.is_cuda:
            sig = sig.cuda()

        if t:
            y = dist.normal(mu2_t1, sig)
        else:
            y = dist.normal(mu2_t0, sig)
        return y


class Encoder(nn.Module):
    def __init__(self, binfeats, contfeats, d, h, nh, activation, cuda):
        super(Encoder, self).__init__()
        self.cuda = cuda
        in_size = binfeats + contfeats
        in2_size = in_size + 1

        # q(t|x)
        self.logits_t = FCNet(in_size, [d], [[1, F.sigmoid]], activation)

        # q(y|x,t)
        self.hqy = FCNet(in_size, (nh - 1) * [h], [], activation)
        self.mu_qy_t0 = FCNet(h, [h], [[1, None]], activation)
        self.mu_qy_t1 = FCNet(h, [h], [[1, None]], activation)

        # q(z|x,t,y)
        self.hqz = FCNet(in2_size, (nh - 1) * [h], [], activation)
        self.muq_t0_sigmaq_t0 = FCNet(
            h, [h], [[d, None], [d, F.softplus]], activation)
        self.muq_t1_sigmaq_t1 = FCNet(
            h, [h], [[d, None], [d, F.softplus]], activation)

    def forward(self, x):
        # q(t|x)
        t_logits = self.logits_t.forward(x)
        t = sample('t', dist.Bernoulli(t_logits).to_event(1))

        # q(y|x,t)
        hqy = self.hqy.forward(x)
        y_loc_t0 = self.mu_qy_t0.forward(hqy)
        y_loc_t1 = self.mu_qy_t1.forward(hqy)
        y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
        y_scale = Variable(torch.ones(y_loc_t0.size()))
        if self.cuda:
            y_scale = y_scale.cuda()
        y = sample('y', dist.Normal(y_loc, y_scale))

        # q(z|x,t,y)
        hqz = self.hqz.forward(torch.cat((x, y), 1))
        z_loc_t0, z_scale_t0 = self.muq_t0_sigmaq_t0.forward(hqz)
        z_loc_t1, z_scale_t0 = self.muq_t1_sigmaq_t1.forward(hqz)
        z_loc = t * z_loc_t1 + (1. - t) * z_loc_t0
        z_scale = t * z_scale_t1 + (1. - t) * z_scale_t0

        return z_loc, z_scale
