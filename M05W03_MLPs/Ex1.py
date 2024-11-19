from util import *
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        out = self.output(x)
        return out.squeeze(1)


class Linear(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x.squeeze(1)


def r_squared(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32).to(device)
    y_pred = torch.tensor(y_pred, dtype=torch.float32).to(device)
    mean_true = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_true) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == "__main__":
    dataset_path = "Auto_MPG_data.csv"
    dataset = pd.read_csv(dataset_path)

    X = dataset.drop(columns='MPG').values
    y = dataset['MPG'].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, random_state=random_state)

    batch_size = 32
    train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = MLP(input_dims=X_train.shape[1], hidden_dims=64, output_dims=1).to(device)

    lr = 1e-2
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs = 100
    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_target = []
        val_target = []
        train_predict = []
        val_predict = []
        model.train()
        for X_samples, y_samples in train_loader:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device)
            optimizer.zero_grad()
            outputs = model(X_samples)
            train_predict += outputs.tolist()
            train_target += y_samples.tolist()
            loss = criterion(outputs, y_samples)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_r2.append(r_squared(train_target, train_predict))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_samples, y_samples in val_loader:
                X_samples = X_samples.to(device)
                y_samples = y_samples.to(device)
                outputs = model(X_samples)
                val_predict += outputs.tolist()
                val_target += y_samples.tolist()
                loss = criterion(outputs, y_samples)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_r2.append(r_squared(val_target, val_predict))

        print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

    model.eval()
    with torch.no_grad():
        y_hat = model(X_test.to(device))
        test_set_r2 = r_squared(y_test, y_hat)
        print('Evaluation on test set:')
        print(f'R2: {test_set_r2}')
