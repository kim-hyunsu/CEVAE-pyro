import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, in_size, layers, out_layers, activation):
        super(FCNet, self).__init__()

        self.activation = activation

           if layers:
                self.input = nn.Linear(in_size, layers[0])
                self.hidden_layers = nn.ModuleList()
                for i in range(1, len(layers)):
                    self.hidden_layers.append(nn.Linear(layers[i], layers[i]))
                self.output_layers = nn.ModuleList()
                self.output_activations = []
                for i, (outdim, activation) in enumerate(out_layers):
                    self.output_layers.append(nn.Linear(layers[-1], outdim))
                    self.output_activations.append(activation)
            else:
                self.output_layers = nn.ModuleList()
                self.output_activations = []
                for i, (outdim, activation) in enumerate(out_layers):
                    self.output_layers.append(nn.Linear(in_size, outdim))
                    self.output_activations.append(activation)

        def forward(self, x):

            x = self.activation(self.input(x) )
            try:
                for layer in self.hidden_layers:
                    x = self.activation(layer(x) )
            except AttributeError:
                pass
            if self.output_layers:
                outputs = []
                for output_layer, output_activation in zip(self.output_layers, self.output_activations):
                    if output_activation:
                        outputs.append(output_activation( output_layer(x) ) )
                    else:
                        outputs.append(output_layer(x) )
                return outputs if len(outputs) > 1 else outputs[0]
            else:
                return x


class Decoder(nn.Module):
    def __init__(self, n_z, nh, h, binfeats, contfeats, activation):
        super(Decoder, self).__init__()

		# p(x|z)
		self.hx = FCNet(n_z, (nh - 1) * [h], [], activation=activation)
		self.logits_1 = FCNet(h, [h], [[binfeats, F.sigmoid]], activation=activation)

		self.mu_sigma = FCNet(h, [h], [[contfeats, None], [contfeats, F.softplus]], activation=activation)

		# p(t|z)
		self.logits_2 = FCNet(n_z, [h], [[1, F.sigmoid]], activation=activation)

		# p(y|t,z)
		self.mu2_t0 = FCNet(n_z, nh * [h], [[1, None]], activation=activation)
		self.mu2_t1 = FCNet(n_z, nh * [h], [[1, None]], activation=activation)

    def forward_P_x(self, z, cont=False):
		# p(x|z)

		hx = self.hx.forward(z)

        if not cont:
            logits = self.logits_1.forward(hx)
            return logits
        else:
            mu, sigma = self.mu_sigma.forward(hx)
            return mu, sigma

    def forward_P_t(self, z):
		# p(t|z)
		logits = self.logits_2(z)
        return logits

    def forward_P_y(self, z, t):
		# p(y|t,z)
		mu2_t0 = self.mu2_t0(z)
		mu2_t1 = self.mu2_t1(z)
        mu2 = t * mu2_t1 + (1. - t) * mu2_t0
		sig = Variable(torch.ones(mu2_t0.size()))

        return mu2, sig

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
    def __init__(self, in_size, in2_size, d, nh, h, binfeats, contfeats, activation):
        super(Encoder, self).__init__()

		# q(t|x)
		self.logits_t = FCNet(in_size, [d], [[1, F.sigmoid]], activation=activation)

		# q(y|x,t)
		self.hqy = FCNet(in_size, (nh - 1) * [h], [], activation=activation)
		self.mu_qy_t0 = FCNet(h, [h], [[1, None]], activation=activation)
		self.mu_qy_t1 = FCNet(h, [h], [[1, None]], activation=activation)

		# q(z|x,t,y)
		self.hqz = FCNet(in2_size, (nh - 1) * [h], [], activation=activation)
		self.muq_t0_sigmaq_t0 = FCNet(h, [h], [[d, None], [d, F.softplus]], activation=activation)
		self.muq_t1_sigmaq_t1 = FCNet(h, [h], [[d, None], [d, F.softplus]], activation=activation)
        
    def forward_Q_t(self, x):
		# q(t|x)
		logits_t = self.logits_t.forward(x)
        
        return logits_t

    def forward_Q_y(self, x, t):
		# q(y|x,t)
		hqy = self.hqy.forward(x)

		mu_qy_t0 = self.mu_qy_t0.forward(hqy)
		mu_qy_t1 = self.mu_qy_t1.forward(hqy)

        mu_qy = t * mu_qy_t1 + (1. - t) * mu_qy_t0

		sig = Variable(torch.ones(mu_qy_t0.size()))

        return mu_qy, sig

    def forward_Q_z(self, x, y, t):
		# q(z|x,t,y)
		hqz = self.hqz.forward(torch.cat((x, y), 1))

		muq_t0, sigmaq_t0 = self.muq_t0_sigmaq_t0.forward(hqz)
		muq_t1, sigmaq_t1 = self.muq_t1_sigmaq_t1.forward(hqz)

        muq = t * muq_t1 + (1. - t) * muq_t0
        sigmaq = t * sigmaq_t1 + (1. - t) * sigmaq_t0

		return muq, sigmaq
