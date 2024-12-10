import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = F.relu(outputs)
        outputs = self.avgpool1(outputs)
        outputs = self.conv2(outputs)
        outputs = F.relu(outputs)
        outputs = self.avgpool2(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc_2(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc_3(outputs)
        return outputs


def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


def evaluate(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VALID_RATIO = 0.9
    BATCH_SIZE = 256
    ROOT = './data'

    train_data = datasets.MNIST(root=ROOT, train=True, download=True)
    test_data = datasets.MNIST(root=ROOT, train=False, download=True)

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(
        train_data,
        [n_train_examples, n_valid_examples]
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

    num_classes = len(train_data.dataset.classes)
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters())

    num_epochs = 10
    save_model = './model'

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        # Training
        train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(lenet_model.state_dict(), save_model + '/lenet_model.pt')
            best_loss_eval = eval_loss

        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

    lenet_model.load_state_dict(torch.load(save_model + '/lenet_model.pt'))
    lenet_model.eval()

    test_data.transform = test_transforms
    test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE)
    test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader, device)
    print(f"Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.3f}")

