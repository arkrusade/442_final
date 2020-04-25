import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from data import initialize_data, data_transforms
from model import Network

from bisect import bisect


class FileManager():
    def __init__(self, data_dir, npy_dir):
        self.data_dir = data_dir
        self.npy_dir = npy_dir
        bound_path = os.path.join(npy_dir, 'header.npy')
        if not os.path.exists(bound_path):
            bounds = []
            count = 0
            for root, dirs, files in os.walk(npy_dir):
                for f in files:
                    path = os.path.join(root, f)
                    data = np.load(path)
                    count = count + data.shape[0]
                    bounds.append(count)
                    print("loaded {}".format(path))

            np.save(bound_path, bounds)

        self.bounds = np.load(bound_path)
        self.bounds.sort()
        self.size = self.bounds[len(self.bounds) - 1]
        self.active_files = {}
        self.queue = []
        self.max_files = 300

    def get_file(self, findex):
        if findex in self.active_files:
            return self.active_files[findex]
        return self.activate_file(findex)

    def activate_file(self, findex):
        data_path = self.npy_dir + '/phd08_data_' + str(findex) + '.npy'
        label_path = self.npy_dir + '/phd08_labels_' + str(findex) + '.npy'
        data_file = np.load(data_path)
        label_file = np.load(label_path)
        tup = (data_file, label_file)
        self.active_files[findex] = tup
        self.queue.append(findex)
        if len(self.active_files) > self.max_files:
            dindex = self.queue.pop(0)
            del self.active_files[dindex]
        return tup

    def get(self, idx):
        assert(idx < self.size)
        findex = bisect(self.bounds, idx)
        data_file, label_file = self.get_file(findex)
        if findex == 0:
            return (data_file[idx], label_file[idx])

        fidx = idx - self.bounds[findex]

        return (data_file[fidx], label_file[fidx])


fm = FileManager("phd08", "phd08_npy_results")
# fm.get(4374)


class PHD_Dataset(Dataset):
    def __init__(self, fm):
        self.fm = fm
        self.mapping = range(fm.size)
        random.seed(442)
        random.shuffle(self.mapping)

    def __len__(self):
        return self.fm.size

    def __getitem__(self, idx):
        return self.fm.get(self.mapping[idx])


allset = PHD_Dataset(fm)
trainset = Subset(allset, range(0, 500000))
valset = Subset(allset, range(500000, 600000))
testset = Subset(allset, range(600000, 700000))

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(valset, batch_size=64, shuffle=False)
# testloader = DataLoader(testset, batch_size=64, shuffle=True)



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

    train(model, train_loader, val_loader, num_epoch)

    vis()

    torch.save(model.state_dict(), "model.pth")
    import pdb; pdb.set_trace()  # XXX BREAKPOINT


if __name__ == '__main__':
    main()
