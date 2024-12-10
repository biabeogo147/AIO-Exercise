import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from Ex1 import LeNetClassifier, train, evaluate


def loader(path):
    return Image.open(path)


if __name__ == "__main__":
    VALID_RATIO = 0.9
    BATCH_SIZE = 256
    data_paths = {
        'train': './train',
        'valid': './validation',
        'test': './test'
    }

    img_size = 150
    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = datasets.ImageFolder(
        root=data_paths['train'],
        loader=loader,
        transform=train_transforms
    )
    valid_data = datasets.ImageFolder(
        root=data_paths['valid'],
        transform=train_transforms
    )
    test_data = datasets.ImageFolder(
        root=data_paths['test'],
        transform=train_transforms
    )

    mean = train_data.dataset.data.float().mean() / 255
    std = train_data.dataset.data.float().std() / 255
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    train_data.dataset.transform = train_transforms
    valid_data.dataset.transform = test_transforms
    train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)

    num_classes = len(train_data.classes)
    lenet_model = LeNetClassifier(num_classes).to(device)

    learning_rate = 2e-4
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters(), learning_rate)

    num_epochs = 10
    save_model = './model'

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, epoch,
                                      log_interval=10)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        if eval_loss < best_loss_eval:
            torch.save(lenet_model.state_dict(), save_model + '/lenet_model.pt')
            best_loss_eval = eval_loss

        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    lenet_model.load_state_dict(torch.load(save_model + '/lenet_model.pt'))
    lenet_model.eval()