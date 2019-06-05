class Inference(object):
    def __init__(self):
        self.test_loader = None
        self.train_loader = None
        self.svi = None
        self.cuda = None

    def train(self):
        epoch_loss = 0.
        for x, _ in self.train_loader:
            if self.cuda:
                x = x.cuda()
            epoch_loss += self.svi.step(x)

        # TODO

    def evaluate(self):
        test_loss = 0.
        for x, _ in self.test_loader:
            if self.cuda:
                x = x.cuda()
            test_loss += self.svi.evaluate_loss(x)

        # TODO
