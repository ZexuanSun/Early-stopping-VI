import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import torch.optim as optim


# Define a basic fully connected neural network for regression tasks using MSE loss
class NN4vi(pl.LightningModule):
    """
    Creates a fully connected neural network.
    
    :param input_dim: int, dimension of input data
    :param hidden_widths: list, size of each hidden layer in the network
    :param output_dim: int, dimension of output
    :param activation: activation function for hidden layers (default is ReLU)
    :param weight_decay: float, weight decay for regularization
    :param lr: float, learning rate for the optimizer
    :return: A network model
    """

    def __init__(self,
                 input_dim: int,
                 hidden_widths: list,
                 output_dim: int,
                 activation=nn.ReLU,  # Use ReLU as the default activation function
                 weight_decay=0,
                 lr=1e-3):
        super().__init__()

        # Define the structure of the neural network
        structure = [input_dim] + list(hidden_widths) + [output_dim]  # Combine input, hidden, and output layers
        layers = []

        # Build the layers of the network
        for j in range(len(structure) - 1):
            act = activation if j < len(structure) - 2 else nn.Identity  # Use Identity for the output layer
            layers += [nn.Linear(structure[j], structure[j + 1]), act()]  # Add Linear layers and activation

        # Use Sequential to create a chain of layers
        self.net = nn.Sequential(*layers)  # Create a sequential model from the layers

        # Store the weight decay and learning rate for use in the optimizer
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the network. Defines how the input moves through the layers.
        
        :param x: input data
        :return: network output
        """
        return self.net(x)  # Pass the input through the network

    def configure_optimizers(self):
        """
        Configures the optimizer for the training process. Uses Adam optimizer.
        
        :return: optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # Set up Adam optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step logic for one batch of data.
        
        :param batch: tuple of (input, target)
        :param batch_idx: index of the current batch
        :return: loss value
        """
        x, y = batch  # Unpack the batch into input and target
        y_hat = self.net(x)  # Forward pass through the network to get predictions
        loss = nn.MSELoss()(y_hat, y)  # Compute Mean Squared Error (MSE) loss

        # Log the training loss to TensorBoard (on step, on epoch, and on the progress bar)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss  # Return the loss value

    def validation_step(self, batch, batch_nb):
        """
        Validation step logic for one batch of validation data.
        
        :param batch: tuple of (input, target)
        :param batch_nb: index of the current validation batch
        :return: validation loss
        """
        x, y = batch  # Unpack the batch
        loss = nn.MSELoss()(self.net(x), y)  # Compute validation loss using MSE
        self.log('val_loss', loss)  # Log the validation loss
        return loss  # Return the validation loss
