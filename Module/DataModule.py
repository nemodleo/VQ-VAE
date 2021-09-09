import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
import os, sys


class CIFAR10_DataModule(pl.LightningDataModule):

    def __init__(self, data_dir='./', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]) # 
        self.cifar_train_variance = 0

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_train, [45000, 5000])
            self.cifar_train_variance = np.var(cifar_train.data / 255.0) # 

        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, pin_memory=True)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_val, batch_size=self.batch_size, pin_memory=True)
        return cifar_val

    def test_dataloader(self):
        cifar_test = DataLoader(self.cifar_test, batch_size=self.batch_size)
        return cifar_test

    def get_cifar_train_variance(self):
        return self.cifar_train_variance