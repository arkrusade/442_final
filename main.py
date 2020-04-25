import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import csv

from torchvision import datasets, transforms
from torch.autograd import Variable

from data import initialize_data, data_transforms
from model import Network


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        # subdirectory for class
        prefix = rootpath + '/' + format(c, '05d') + '/'
        filepath = prefix + 'GT-'+ format(c, '05d') + '.csv'
        gtFile = open(filepath)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
        print("finished loading class {}".format(c))
    return images, labels


device = "cuda" if torch.cuda.is_available() else "cpu"  # Configure device
criterion = nn.CrossEntropyLoss()  # Specify the loss layer
model = Network().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
num_epoch = 10


def train(model, loader, valloader, num_epoch=10):  # Train the model
    print("Start training...")
    model.train()  # Set the model to training mode
    losses = []
    val = []
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            pred = model(batch)  # This will call Network.forward() that you implement
            loss = criterion(pred, label)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
        print("Epoch {} loss:{}".format(i+1, np.mean(running_loss)))  # Print the average loss for this epoch
        print("Evaluate on validation set...")
        val.append(evaluate(model, valloader))
        losses.append(running_loss)
    print("Done!")
    np.save("loss.npy", losses)
    np.save("val.npy", val)


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc


def vis():
    loss = np.load("loss.npy")
    val = np.load("val.npy")
    loss = np.mean(loss, axis=1)
    plt.plot(loss, label="Loss")
    plt.plot(val, label="Validation accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("loss.png")
    plt.close()


def main():
    rootdir = "."
    batch_size = 64
    initialize_data(rootdir) # extracts the zip files, makes a validation set

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(rootdir + '/train_images',
                            transform=data_transforms),
        batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(rootdir + '/val_images',
                            transform=data_transforms),
        batch_size=batch_size, shuffle=False, num_workers=1)

    train(model, train_loader, val_loader, num_epoch)

    vis()

    torch.save(model.state_dict(), "model.pth")
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    # path = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    path = './GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
    images, labels = readTrafficSigns(path)


if __name__ == '__main__':
    main()
