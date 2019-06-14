import argparse
from inference import Inference
import torch
from pyro.optim import Adam
import numpy as np

if __name__ == "__main__":
    # Command line
    parser = argparse.ArgumentParser(description="CEVAE-Pyro")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--l2-weight-decay', type=float, default=0.0001)
    parser.add_argument('--z-dim', type=int, default=20)
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--data-file', choices=['npz', 'csv'], default='csv')
    args = parser.parse_args()

    # Data
    if args.data_file != 'csv':
        from data_loader import IHDPNPZDataset, IHDPNPZDataLoader
        path = 'data/IHDP-np'
        test_data = IHDPNPZDataset(np.load(f'{path}/ihdp_npci_1-100.test.npz'))
        train_data = IHDPNPZDataset(
            np.load(f'{path}/ihdp_npci_1-100.train.npz'))
        data_loader = IHDPNPZDataLoader(test_data, train_data)
        binary_indices, continuous_indices = train_data.indices_each_features()

    else:
        from data_loader import IHDPDataset, IHDPDataLoader
        dataset = IHDPDataset('data/IHDP')
        binary_indices, continuous_indices = dataset.indices_each_features()
        data_loader = IHDPDataLoader(dataset, validation_split=0.1)

    train_loader, test_loader = data_loader.loaders(batch_size=args.batch_size)

    # CEVAE
    cuda = torch.cuda.is_available()
    print(f"CUDA: {cuda}")
    optimizer = Adam({
        "lr": args.learning_rate, "weight_decay": args.l2_weight_decay
    })
    activation = torch.nn.functional.elu
    inference = Inference(len(binary_indices), len(continuous_indices), args.z_dim,
                          args.hidden_dim, args.hidden_layers, optimizer, activation, cuda)

    # Training
    train_elbo = []
    test_elbo = []
    for epoch in range(args.epochs):
        total_epoch_loss_train = inference.train(train_loader)
        train_elbo.append(-total_epoch_loss_train)
        print(
            f"[epoch {epoch:03d}] average training loss: {total_epoch_loss_train:.4f}")
        if epoch % 5 == 0:
            total_epoch_loss_test = inference.evaluate(test_loader)
            test_elbo.append(-total_epoch_loss_test)
            print(
                f"[epoch {epoch:03d}] average test loss: {total_epoch_loss_test:.4f}")
