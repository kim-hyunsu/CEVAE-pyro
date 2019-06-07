import argparse
from data_loader import IHDPLoader
from inference import Inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEVAE-Pyro")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--z-dim', type=int, default=20)
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=200)

    data = IHDPLoader()

    binary_indices, continuous_indices = data.indices_binary_continuous()
    train_loader, test_loader = data.loaders(batch_size=128)

    inference = Inference()  # TODO

    inference.train(train_loader)
