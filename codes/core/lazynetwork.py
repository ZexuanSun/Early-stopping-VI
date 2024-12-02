import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
import collections
from itertools import repeat
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torch.optim as optim

# Function to center and scale the output of a given function using mean and standard deviation
def center_function(f, mult=1.0):
    y = f(normal_sampling)  # Apply function f to normal_sampling
    mean = torch.mean(y).item()  # Calculate mean of the output
    stddev = torch.sqrt(torch.mean((y - mean) ** 2)).item()  # Calculate standard deviation
    print(mean, stddev)  # Print mean and standard deviation
    mult /= stddev  # Adjust scaling factor based on standard deviation
    # Inner function to apply centering and scaling
    def ff(x):
        return (f(x) - mean) * mult  # Center and scale function output
    return ff  # Return the centered and scaled function

# Utility function to create a tuple of size n
def _ntuple(n):
    def parse(x):
        # Return a tuple if x is iterable; otherwise, create a tuple of n identical elements
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

# Creating utility functions for single, pair, triple, and quadruple ntuples
_single = _ntuple(1)  # Utility for single values
_pair = _ntuple(2)    # Utility for pairs
_triple = _ntuple(3)  # Utility for triples
_quadruple = _ntuple(4)  # Utility for quadruples

# Function to compute the double factorial of a number
def double_factorial(n):
    if n <= 1:
        return 1  # Base case for double factorial
    else:
        return n * double_factorial(n - 2)  # Recursive case for double factorial

# Custom linear layer with configurable bias and beta factor
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
        super(Linear, self).__init__()  # Initialize the parent class
        self.in_features = in_features  # Number of input features
        self.out_features = out_features  # Number of output features
        # Initialize weights with or without CUDA
        self.weight = Parameter(torch.Tensor(out_features, in_features).cuda()) if torch.cuda.is_available() else Parameter(torch.Tensor(out_features, in_features))
        
        self.stdv = np.sqrt(1 - beta**2) / math.sqrt(in_features * 1.0)  # Standard deviation for scaling
        self.beta = beta  # Scaling factor for bias

        # Initialize bias if specified
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).cuda()) if torch.cuda.is_available() else Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)  # No bias if not specified
        
        self.reset_parameters()  # Initialize weights and bias

    # Method to reset parameters, with optional zero initialization
    def reset_parameters(self, zero=False):
        if not zero:
            # Initialize weights and bias with normal distribution
            self.weight.data.normal_(0.0, 1.0)
            if self.bias is not None:
                self.bias.data.normal_(0.0, 1.0)
        else:
            # Zero initialize weights and bias
            torch.nn.init.zeros_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    # Forward pass through the custom linear layer
    def forward(self, input):
        if self.bias is not None:
            # Apply scaling with bias
            return F.linear(input, self.weight) * self.stdv + self.bias * self.beta
        else:
            # Apply scaling without bias
            return F.linear(input, self.weight) * self.stdv

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



# Custom PyTorch Lightning module
class LazyNet(pl.LightningModule):
    def __init__(self, widths, non_lin=torch.relu, bias=True, beta=0.1, weight_decay=0, lr=1, batch_norm=False):
        super(LazyNet, self).__init__()  # Initialize the parent class
        self.widths = widths  # List of layer sizes
        self.depth = len(self.widths) - 1  # Determine the depth of the network
        self.non_lin = non_lin  # Set non-linear activation function
        self.beta = beta  # Set scaling factor for bias
        self.batch_norm = batch_norm  # Flag to determine if batch normalization is used
        self.weight_decay = weight_decay  # Weight decay for regularization in optimizer
        self.lr = lr  # Learning rate for optimizer

        self.pre_alpha = [None for _ in range(self.depth)]  # Pre-activation values for each layer
        self.alpha = [None for _ in range(self.depth)]  # Activation values for each layer

        # Initialize linear layers
        self.linears = []
        for i in range(self.depth):
            lin = Linear(widths[i], widths[i+1], bias, beta)  # Create a Linear layer
            self.add_module('lin' + str(i).zfill(2), lin)  # Register layer with a name
            self.linears.append(lin)  # Add layer to the list

        # Initialize batch normalization layers if enabled
        if self.batch_norm:
            self.bns = []  # List to hold batch normalization layers
            for i in range(1, self.depth):
                bn = BatchNorm1d(widths[i], affine=False, eps=0.1)  # Create BatchNorm layer
                self.add_module('bn' + str(i).zfill(2), bn)  # Register BatchNorm layer with a name
                self.bns.append(bn)  # Add to the list of BatchNorm layers

    # Method to reset parameters for each layer
    def reset_parameters(self, zero=False):
        for l in self.linears:
            l.reset_parameters(zero=zero)  # Reset parameters for each linear layer

    # Initialize parameters with another model's parameters
    def init_parameters(self, full_nn):
        params_full = tuple(full_nn.parameters())  # Get parameters from the full network
        for k, param in enumerate(self.parameters()):
            param.data = params_full[k].clone()  # Clone parameters into the current model

    # Forward pass through the network
    def forward(self, x):
        self.alpha[0] = x  # Set the input as the first activation value
        for i in range(self.depth - 1):
            self.pre_alpha[i + 1] = self.linears[i](self.alpha[i])  # Apply linear transformation
            self.alpha[i + 1] = self.non_lin(self.pre_alpha[i + 1])  # Apply non-linearity
            if self.batch_norm:
                self.alpha[i + 1] = self.bns[i](self.alpha[i + 1])  # Apply batch normalization if enabled
        return self.linears[self.depth - 1](self.alpha[self.depth - 1])  # Final layer output

    # Compute covariance matrix
    def Sigma(self, i):
        return torch.matmul(self.alpha[i - 1], torch.t(self.alpha[i - 1])) / self.widths[i - 1] + self.beta ** 2

    # Compute first derivative of the covariance matrix
    def Sigma_dot(self, i, retain_graph=False):
        alpha_dot = torch.autograd.grad(self.alpha[i - 1].sum(), self.pre_alpha[i - 1], retain_graph=retain_graph)[0]
        return torch.matmul(alpha_dot, torch.t(alpha_dot)) / self.widths[i - 1]

    # Compute second derivative of the covariance matrix
    def Sigma_ddot(self, i, retain_graph=False):
        alpha_dot = torch.autograd.grad(self.alpha[i - 1].sum(), self.pre_alpha[i - 1], create_graph=True)[0]
        alpha_ddot = torch.autograd.grad(alpha_dot.sum(), self.pre_alpha[i - 1], retain_graph=retain_graph)[0]
        return torch.matmul(alpha_ddot, torch.t(alpha_ddot)) / self.widths[i - 1]

    # Neural Tangent Kernel (NTK) computation
    def NTK(self, retain_graph=False):
        K = self.Sigma(1)  # Initialize NTK with the covariance matrix of the first layer
        for i in range(1, self.depth):
            K = K * self.Sigma_dot(i + 1, retain_graph) + self.Sigma(i + 1)  # Update NTK
        return K  # Return the computed NTK

    # Compute various moment statistics
    def moments_S(self):
        NTK = self.Sigma(1)  # Compute NTK for the first layer
        m2 = 0  # Initialize second moment
        mom1 = self.alpha[1].clone().zero_()  # Initialize first moment

        # Initialize tensors for moments
        if torch.cuda.is_available():
            covar_m1 = torch.zeros([1, 1]).cuda()  # Use GPU if available
            move_m1 = torch.zeros([1, 1]).cuda()
        else:
            covar_m1 = torch.zeros([1, 1])  # Use CPU
            move_m1 = torch.zeros([1, 1])

        for j in range(1, self.depth):
            alpha_dot = torch.autograd.grad(self.alpha[j].sum(), self.pre_alpha[j], create_graph=True)[0]
            alpha_ddot = torch.autograd.grad(alpha_dot.sum(), self.pre_alpha[j], create_graph=True)[0]
            alpha_dddot = torch.autograd.grad(alpha_ddot.sum(), self.pre_alpha[j], retain_graph=True)[0]

            Sigma = self.Sigma(j + 1)  # Compute covariance matrix for current layer
            Sigma_dot = torch.matmul(alpha_dot, torch.t(alpha_dot)) / self.widths[j]  # First derivative
            Sigma_ddot = torch.matmul(alpha_ddot, torch.t(alpha_ddot)) / self.widths[j]  # Second derivative
            Mixed = torch.matmul(alpha_ddot, torch.t(self.alpha[j])) / self.widths[j]  # Mixed moments
            Mixed_dot = torch.matmul(alpha_dddot, torch.t(alpha_dot)) / self.widths[j]  # Mixed dot moments

            # Update second moment statistics
            m2 = m2 * Sigma_dot + NTK * NTK * Sigma_ddot + 2 * NTK * Sigma_dot
            move_m1 = move_m1 * Sigma_dot
            move_m1 += NTK * (covar_m1.diag().view([-1, 1]) * Mixed_dot + covar_m1 * Sigma_ddot)
            move_m1 += covar_m1.diag().view([-1,

