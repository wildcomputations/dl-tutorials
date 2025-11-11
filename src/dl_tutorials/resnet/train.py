#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.datasets import MNIST

# model
# - normalized initialization (to prevent vanishing/exploding gradient)
# - intermediate normalization


# TODO list
# - padding to 32x32 images
# - tensorboard reporting
# - get it to run
# - param initialization
# - hyperparam generaliztion
# - check with chatGPT how this person on the internet didn't follow resnet
#   principles
# - documentation

# maybes?
# - different initialization
# - different learning rate schedules
# - weight decay


class ResidualLayer(nn.Module):
    """From the https://arxiv.org/abs/1603.05027 follow on paper to resnet"""
    def __init__(self, conv_size: int, num_ch: int, stride: int = 1):
        super().__init__(self)
        self.residual_mapping = nn.Sequential(
            nn.BatchNorm2d(num_ch),
            nn.ReLU(),
            nn.Conv2d(num_ch, num_ch, conv_size),
            nn.BatchNorm2d(num_ch),
            nn.ReLU(),
            nn.Conv2d(num_ch, num_ch, conv_size, stride=stride),
        )
        self.projection: Callable[[torch.Tensor], torch.Tensor]
        if stride != 1:
            self.projection = nn.Conv2d(num_ch, num_ch * 2, 1, stride)
        else:
            self.projection = lambda x: x

        # TODO normalization

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, C, H, W)
        """
        x_res = self.residual_mapping(x)
        x = x_res + self.projection(x)
        return x


class MiniResNet(nn.Module):
    def __init__(self, n_classes: int):  #? change to hyparams? Assuming 32x32 monochrome input
        self.layers = nn.Sequential(
            ResidualLayer(7, 1, 2),  # B, C=2, H=16, W=16

            ResidualLayer(3, 2),  # B, C=2, H=16, W=16
            ResidualLayer(3, 2),  # B, C=2, H=16, W=16
            ResidualLayer(3, 2, 2),  # B, C=4, H=8, W=8

            ResidualLayer(3, 4),  # B, C=4, H=8, W=8
            ResidualLayer(3, 4),  # B, C=4, H=8, W=8
            ResidualLayer(3, 4, 2),  # B, C=8, H=4, W=4

            # Now flatten
            nn.AvgPool2d(4),  # B, C=8

            # and a fully connected layer
            nn.Linear(8, n_classes)
        )


class MnistClassifyModule(L.LightningModule):
    """
    Train a network to classify the number in MNIST hand writing examples.
    """
    def __init__(self):
        super().__init__(self)
        self.save_hyperparameters()

        self.model = MiniResNet(10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        return nn.functional.softmax(self.model(batch), dim=-1)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss(output, target)
        return loss


class MnistDataModule(L.LightningDataModule):
    """MNIST dataset. Images are 28x28 pixels"""
    def __init__(self, root_dir: Path, batch_size: int):
        super().__init__()
        # if we restart from a checkpoint, use the same root_dir and batch_size
        # The default behavior of `save_hyperparameters` is to save all the
        # arguments to the current method.
        # See https://www.codegenes.net/blog/pytorch-lightning-save_hyperparameters/
        # for some good examples
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.batch_size = batch_size

        # depending on the stage, we might instantiate different datasets
        self.mnist_train: Dataset
        self.mnist_val: Dataset
        self.mnist_test: Dataset

        # TODO: learn proper torchvision transformers
        self.pad = torchvision.transforms.Pad(2)

    def prepare_data(self):
        """One time download at beginning of process"""
        # download train and test sets
        MNIST(self.root_dir, train=True, download=True)
        MNIST(self.root_dir, train=False, download=True)

    def setup(self, stage: str):
        """Create necessary datasets"""
        if stage in ["fit", "validate"]:
            # Split train vs validation in setup() before the trainer moves the
            # datamodule to the GPUs. This way all GPUs have consistent data sets

            mnist_full = MNIST(self.root_dir, train=True) # TODO transform=self.transform)
            # Use a fixed seed so that variation between loss is entirely due
            # to model changes, not data variability.
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [0.6, 0.4], generator=torch.Generator().manual_seed(42)
            )
        elif stage == "test":
            # Only need the test set if the stage is testing
            self.mnist_test = MNIST(self.root_dir, train=False) # TODO transforms

    def train_dataloader(self):
        """Get the training data"""
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Get the training data"""
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Get the training data"""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path,
                        default=Path.home() / "data" / "mnist",
                        help="directory to cache dataset")
    # TODO: checkpoints
    # TODO: train vs test
    return parser.parse_args()


def main():
    """
    Either train or test a model depending on the command line arguments
    """
    args = parse_args()

    data_module = MnistDataModule(args.dataset_root, 32)

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{step}",
        save_top_k=-1,  # keep all checkpoints
        every_n_epochs=1,
    )

    trainer = L.Trainer(callbacks=[checkpoint_cb])

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
