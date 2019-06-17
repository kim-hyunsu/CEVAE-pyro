import argparse
from data_loader import IHDPDataset, IHDPDataLoader
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
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--z-dim', type=int, default=20)
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=200)
    args = parser.parse_args()

    # Data
    path = 'data/IHDP'
    data = np.concatenate([np.loadtxt(
        f"{path}/ihdp_npci_{index+1}.csv", delimiter=',') for index in range(10)], 0)
    dataset = IHDPDataset(data)
    y_mean, y_std = dataset.y_mean_std()
    binary_indices, continuous_indices = dataset.indices_each_features()
    data_loader = IHDPDataLoader(dataset, validation_split=0.1)

    train_loader, test_loader = data_loader.loaders(batch_size=args.batch_size)

    # CEVAE
    cuda = torch.cuda.is_available()
    print(f"CUDA: {cuda}")
    optimizer = Adam({
        "lr": args.learning_rate, "weight_decay": args.weight_decay
    })
    activation = torch.nn.functional.elu
    inference = Inference(y_mean, y_std, len(binary_indices), len(continuous_indices), args.z_dim,
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

            (ITE, ATE, PEHE), (RMSE_factual,
                               RMSE_counterfactual) = inference.train_statistics(L=1)
            print(f"[epoch {epoch:03d}] ITE: {ITE:0.3f}, ATE: {ATE:0.3f}, PEHE: {PEHE:0.3f}, Factual RMSE: {RMSE_factual:0.3f}, Counterfactual RMSE: {RMSE_counterfactual:0.3f}")
