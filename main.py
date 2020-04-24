from __future__ import print_function
import matplotlib.pyplot as plt
import csv

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from data import initialize_data, data_transforms


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

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    # path = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    path = './GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
    images, labels = readTrafficSigns(path)


if __name__ == '__main__':
    main()
