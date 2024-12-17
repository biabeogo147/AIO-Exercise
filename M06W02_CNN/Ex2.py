from M06W02_CNN.util import *


class ScenesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x.clone().detach()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.cat([res, x], 1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BottleneckBlock(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_blocks = nn.ModuleList()
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(num_layers, in_channels, growth_rate))
            in_channels += num_layers * growth_rate
            if i != len(num_blocks) - 1:
                out_channels = in_channels // 2
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                ))
                in_channels = out_channels

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.pool2 = nn.AvgPool2d(kernel_size=7)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.dense_blocks:
            x = block(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    seed = 59
    set_seed(seed)

    root_dir = 'scenes_classification'
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'val')

    classes = {
        label_idx: class_name
        for label_idx, class_name in enumerate(
            sorted(os.listdir(train_dir))
        )
    }

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for dataset_path in [train_dir, test_dir]:
        for label_idx, class_name in classes.items():
            class_dir = os.path.join(dataset_path, class_name)
            for img_filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_filename)
                if 'train' in dataset_path:
                    X_train.append(img_path)
                    y_train.append(label_idx)
                else:
                    X_test.append(img_path)
                    y_test.append(label_idx)

    seed = 0
    val_size = 0.2
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    train_dataset = ScenesDataset(
        X_train, y_train,
        transform=transform
    )
    val_dataset = ScenesDataset(
        X_val, y_val,
        transform=transform
    )
    test_dataset = ScenesDataset(
        X_test, y_test,
        transform=transform
    )

    train_batch_size = 64
    test_batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    n_classes = len(list(classes.keys()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DenseNet(
        num_blocks=[6, 12, 24, 16],
        growth_rate=32,
        num_classes=n_classes
    ).to(device)

    lr = 1e-2
    epochs = 15

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print('Evaluation on val / test dataset')
    print('Val accuracy:', val_acc)
    print('Test accuracy:', test_acc)