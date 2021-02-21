import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .unet_utils import Down, Up, DoubleConv


class LitUNet(pl.LightningModule):
    def __init__(
            self,
            num_classes: int,
            *,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,

    ):
        """
        Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
        <https://arxiv.org/abs/1505.04597>`_
        Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
        Implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_
            - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
        Args:
            num_classes: Number of output classes required
            num_layers: Number of layers in each side of U-net (default 5)
            features_start: Number of features in first layer (default 64)
            bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
        """
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        # TODO: change the convolution so that it predicts masks and not classes
        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]

        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        pred = self.layers[-1](xi[-1])

        return pred

    @staticmethod
    def calculate_loss(logits, y):
        # TODO: Somehow, calculate loss using Low Entropy Motion Loss
        loss = ...
        return loss

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)

        loss = self.calculate_loss(logits, y)

        self.log('train_loss', loss)
        return loss

    def general_step(self, step_name, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)

        self.log(step_name + '_loss', loss, prog_bar=True)

        # TODO: Calculate accuracy maybe ?
        # acc = ...
        # self.log(step_name + '_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        self.general_step('val', batch, batch_nb)

    def test_step(self, batch, batch_nb):
        self.general_step('test', batch, batch_nb)

    def configure_optimizers(self):
        # TODO: Find some optimizer
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        pass

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # TODO: load train data
            data_full = ...
            train_size = int(len(data_full) * 0.9)
            val_size = len(data_full) - train_size

            self.data_train, self.data_val = random_split(data_full, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # TODO: load test data
            self.data_test = ...

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
