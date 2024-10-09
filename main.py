# Standard library imports
import numpy as np
import matplotlib.pyplot as plt

# Third-party imports
import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F

def main():
    image_path = './'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_dataset = torchvision.datasets.MNIST(
        root=image_path, train=True,
        transform=transform, download=True
    )

    mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))

    mnist_test_dataset = torchvision.datasets.MNIST(
        root=image_path, train=False,
        transform=transform, download=False
    )

    batch_size = 64
    torch.manual_seed(1)
    train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

    model = nn.Sequential()
    model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2))

    model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

    """Checking the output size after convolutions and pooling.
    We need to know whats the output of all convolutions and pooling layers to know the input to Feed Forward MLP's,
    Popular method to calculate the output size is calculating the two formulas for each convolution and pooling:
    for convolution:
        o = (floor(n+2p-m/s)) + 1, where:
            floor - denotes flooring operation
            n - spacial dimension of the input
            p - padding
            m - kernel size
            s - stride
    for pooling:
        o = (floor(n - k)) + 1, where:
            n - spacial dimension of the input to the pooling layer
            k - kernel_size
            
    but a lot more practical is creating a dummy in a shape of our input tensor,
    feed that to our model, and check the shape. In our case its torch.Size([64, 64, 7, 7])
     """
    dummy_tensor = torch.ones(64, 1, 28, 28)
    print(model(dummy_tensor).shape)
    """We add nn.Flatten() to reshape our tensor back to 2D"""
    model.add_module('flatten', nn.Flatten())

    model.add_module('fc1', nn.Linear(3136, 1024))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))

    model.add_module('fc2', nn.Linear(1024, 10))

    # Save the cross-entropy loss function as a criterion
    criterion = F.cross_entropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train(model, num_epochs, train_dl, valid_dl):
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs
        for epoch in range(num_epochs):
            model.train()
            for x_batch, y_batch in train_dl:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Update the total loss for this epoch.
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                """torch.argmax(pred, dim=1) finds the index of the maximum value along dimension 1, which is across columns
                 for each example in the batch.
                In the context of a tensor with shape [batch_size, num_classes] (e.g., [64, 10]), it returns the index
                 of the maximum value in each row (i.e., across columns),
                  giving you the predicted class for each example in the batch."""
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum().item()
                # Normalizes total loss to average loss per sample:
                # (len(train_dl) - number of batches,  # (len(train_dl.dataset) - number of samples.
            loss_hist_train[epoch] /= len(train_dl.dataset)
            accuracy_hist_train[epoch] /= len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)

                    # Update the total loss for this epoch
                    loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                    is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                    accuracy_hist_valid[epoch] += is_correct.sum()
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

            print(f' Epoch {epoch+1} accuracy: '
                  f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
                  f'{accuracy_hist_valid[epoch]:.4f}')

        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

    # Call train function
    torch.manual_seed(1)
    num_epochs = 20
    hist = train(model, num_epochs, train_dl, valid_dl)


    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_arr, hist[0], '-o', label='Train loss')
    ax1.plot(x_arr, hist[1], '--<', label='Validation loss')
    ax1.legend(fontsize=15)
    ax1.set_xlabel('Epoch', size=15)
    ax1.set_ylabel('Loss', size=15)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(x_arr, hist[2], '-o', label='Train acc')
    ax2.plot(x_arr, hist[3], '--<', label='Validation acc')
    ax2.legend(fontsize=15)
    ax2.set_xlabel('Epoch', size=15)
    ax2.set_ylabel('Accuracy', size=15)

    plt.show()

if __name__ == "__main__":
    main()