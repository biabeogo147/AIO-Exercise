from util import *
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.output(x)
        return out.squeeze(1)


def compute_accuracy(y_hat, y_true):
    _, y_hat = torch.max(y_hat, dim=1)
    correct = (y_hat == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy


if __name__ == "__main__":
    data_path = 'NonLinear_data.npy'
    data = np.load(data_path, allow_pickle=True).item()
    X, y = data['X'], data['labels']
    print(X[:5], np.unique(y))
    print(X.shape, y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, random_state=random_state)

    batch_size = 32
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = MLP(
        input_dims=X_train.shape[1],
        hidden_dims=512,
        output_dims=torch.unique(y_train).shape[0]
    ).to(device)

    lr = 1e-1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs = 100
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        train_loss = 0.0
        train_target = []
        train_predict = []
        model.train()

        for X_samples, y_samples in train_loader:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device).long()
            optimizer.zero_grad()
            outputs = model(X_samples)
            loss = criterion(outputs, y_samples)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_predict.append(outputs.detach().cpu())
            train_target.append(y_samples.cpu())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        train_predict = torch.cat(train_predict)
        train_target = torch.cat(train_target)
        train_acc = compute_accuracy(train_predict, train_target)
        train_accs.append(train_acc)

        val_loss = 0.0
        val_target = []
        val_predict = []

        model.eval()
        with torch.no_grad():
            for X_samples, y_samples in val_loader:
                X_samples = X_samples.to(device)
                y_samples = y_samples.to(device).long()
                outputs = model(X_samples)
                val_loss += criterion(outputs, y_samples).item()

                val_predict.append(outputs.cpu())
                val_target.append(y_samples.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_predict = torch.cat(val_predict)
        val_target = torch.cat(val_target)
        val_acc = compute_accuracy(val_predict, val_target)
        val_accs.append(val_acc)

        print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].plot(train_losses, color='green')
    ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
    ax[0, 0].set_title('Training Loss')

    ax[0, 1].plot(val_losses, color='orange')
    ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
    ax[0, 1].set_title('Validation Loss')

    ax[1, 0].plot(train_accs, color='green')
    ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy')
    ax[1, 0].set_title('Training Accuracy')

    ax[1, 1].plot(val_accs, color='orange')
    ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy')
    ax[1, 1].set_title('Validation Accuracy')

    plt.show()

    test_target = []
    test_predict = []
    model.eval()

    with torch.no_grad():
        for X_samples, y_samples in test_loader:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device)
            outputs = model(X_samples)

            test_predict.append(outputs.cpu())
            test_target.append(y_samples.cpu())

    test_predict = torch.cat(test_predict)
    test_target = torch.cat(test_target)
    test_acc = compute_accuracy(test_predict, test_target)

    print('Evaluation on test set:')
    print(f'Accuracy: {test_acc}')
