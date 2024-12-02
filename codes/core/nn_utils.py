
from itertools import chain, combinations
import torch
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import os
from pytorch_lightning import Callback
from pytorch_lightning import trainer, LightningModule
from sklearn.model_selection import KFold
from lazynetwork import *
import matplotlib.pyplot as plt

def vi_exp_wrapper(X, Y, drop_i, widths, lr=0.1, eslr=0.1):
    # Split data into training, validation, and test sets
    X_fit, X_test, y_fit, y_test = train_test_split(X, Y, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit, random_state=1)

    # Create a copy of the training data and drop the specified feature
    X_fit_drop = X_fit.clone()
    X_fit_drop[:, drop_i] = torch.mean(X_fit_drop[:, drop_i])

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Drop the specified feature from the training, validation, and test sets
    X_train_drop, X_val_drop, X_test_drop = dropdata(X_train, X_val, X_test, drop_i)

    # Initialize data module for training
    dm = FlexDataModule(X_train, y_train, X_val, y_val)

    # Initialize and train the full neural network
    full_nn = LazyNet(widths, lr=lr)
    full_nn.reset_parameters()

    # Set up early stopping and metric tracking
    early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
    cb = MetricTracker()
    trainer = pl.Trainer(callbacks=[cb, early_stopping], max_epochs=800, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(full_nn, dm)
    print(trainer.current_epoch)

    # Initialize lazy neural network and transfer parameters from the full network
    lazy_nn = LazyNet(widths, lr=eslr)
    lazy_nn.init_parameters(full_nn)

    # Initialize data module for training with dropped data
    dm_lazy = FlexDataModule(X_train_drop, y_train, X_val_drop, y_val)
    early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
    cb = MetricTracker()
    trainer = pl.Trainer(callbacks=[cb, early_stopping], max_epochs=100, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(lazy_nn, dm_lazy)
    print(trainer.current_epoch)

    # Calculate and print variational inference estimates
    vi_est = (torch.mean((lazy_nn(X_test_drop) - y_test) ** 2) - torch.mean((full_nn(X_test) - y_test) ** 2)).item()

    # Initialize and train a reduced neural network
    red_nn = LazyNet(widths, lr=lr)
    red_nn.reset_parameters()

    # Initialize data module for training with dropped data
    dm_red = FlexDataModule(X_train_drop, y_train, X_val_drop, y_val)
    early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
    cb = MetricTracker()
    trainer = pl.Trainer(callbacks=[cb, early_stopping], max_epochs=400, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(red_nn, dm_red)
    print(trainer.current_epoch)

    # Calculate additional variational inference estimates
    vi_retrain = (torch.mean((red_nn(X_test_drop) - y_test) ** 2) - torch.mean((full_nn(X_test) - y_test) ** 2)).item()
    vi_drop = (torch.mean((full_nn(X_test_drop) - y_test) ** 2) - torch.mean((full_nn(X_test) - y_test) ** 2)).item()
    
    # Print variational inference results
    print('es', vi_est)
    print('retrain', vi_retrain)
    print('drop', vi_drop)

    return vi_est, vi_retrain, vi_drop

def powerset(iterable):
    """Generate the powerset of a given iterable.
    
    Args:
        iterable: An iterable object (e.g., list, set).
        
    Returns:
        A generator that yields subsets of the iterable.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def dropdata(X_train, X_val, X_test, dropi):
    """Drop a feature from training, validation, and test sets.
    
    Args:
        X_train: Training data.
        X_val: Validation data.
        X_test: Test data.
        dropi: Index of the feature to drop.
        
    Returns:
        Tuple containing the modified training, validation, and test sets.
    """
    X_train_drop = X_train.clone()
    X_val_drop = X_val.clone()
    X_test_drop = X_test.clone()

    # Replace the dropped feature with its mean
    X_train_drop[:, dropi] = torch.mean(X_train_drop[:, dropi], axis=0)
    X_test_drop[:, dropi] = torch.mean(X_test_drop[:, dropi], axis=0)
    X_val_drop[:, dropi] = torch.mean(X_val_drop[:, dropi], axis=0)

    return X_train_drop, X_val_drop, X_test_drop


def generate_WV(beta, m, V='random', sigma=0.1):
    """Generate weight matrix W and vector V based on parameters.

    Args:
        beta: Coefficients to define the mean of the normal distribution for W.
        m: Number of samples to generate.
        V: Either 'random' for random values or 'sequential' for evenly spaced values.
        sigma: Standard deviation for the normal distribution used to generate W.

    Returns:
        W: Generated weight matrix of shape (m, len(beta)).
        V: Generated vector based on the specified type ('random' or 'sequential').
    """
    p = len(beta)  # Number of parameters
    W = np.zeros((m, p))  # Initialize W as a zero matrix

    # Generate W from a normal distribution centered around each beta
    for j, b in enumerate(beta):
        W[:, j] = np.random.normal(b, sigma, size=m)

    W = torch.tensor(W, dtype=torch.float32)  # Convert W to a tensor

    # Generate V based on the specified type
    if V == 'random':
        V = torch.tensor(np.random.normal(size=(1, m)), dtype=torch.float32)
    elif V == 'sequential':
        V = torch.tensor((np.arange(m) + 1) / m, dtype=torch.float32)

    return W, V

def generate_2lnn_data(W, V, n, corr=0.5):
    """Generate data using a two-layer neural network structure.

    Args:
        W: Weight matrix from the first layer.
        V: Weight vector from the second layer.
        n: Number of samples to generate.
        corr: Correlation coefficient for the input features.

    Returns:
        X: Input features as a tensor.
        Y: Output labels as a tensor.
    """
    p = W.shape[1]  # Number of features
    sigma = np.eye(p)  # Initialize covariance matrix as an identity matrix
    sigma[0, 1] = corr  # Set correlation
    sigma[1, 0] = corr

    # Generate input features from a multivariate normal distribution
    X = np.random.multivariate_normal(np.zeros(p), sigma * 0.1, size=n)
    X = torch.tensor(X, dtype=torch.float32)  # Convert X to a tensor

    # Compute output labels using the neural network structure
    Y = torch.tensor(torch.matmul(V, torch.relu(torch.matmul(W, X.T))).detach().numpy(),
                     dtype=torch.float32)

    return X, Y.reshape(-1, 1)  # Reshape Y to be a column vector

def generate_logistic_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    """Generate data for logistic regression.

    Args:
        beta: Coefficients for the logistic regression.
        sigma: Standard deviation for the noise added to the output.
        N: Number of samples to generate.
        seed: Seed for random number generation.
        corr: Correlation coefficient for the input features.

    Returns:
        X: Input features as a tensor.
        Y: Output labels as a tensor.
    """
    random.seed(seed)  # Set the random seed for reproducibility
    cov = [[1, corr], [corr, 1]]  # Define covariance for correlated features
    beta = np.array(beta, dtype=float)
    p = beta.shape[0]  # Number of parameters

    # Generate normally distributed input features
    X = np.random.normal(0, 1, size=(N, p))
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)

    normal_noise = np.random.normal(0, sigma, size=N)  # Generate noise
    EY = 1 / (1 + np.exp(-X @ beta))  # Compute expected values for logistic function
    Y = EY + normal_noise  # Add noise to expected values

    X = torch.tensor(X, dtype=torch.float32)  # Convert X to a tensor
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)  # Convert Y to a tensor and reshape

    return X, Y

def create_gen(beta, widths, N=100, corr=0.0):
    """Create and train a generative model based on the specified parameters.

    Args:
        beta: Coefficients for generating data.
        widths: Layer widths for the neural network.
        N: Number of samples to generate.
        corr: Correlation for the input features.

    Returns:
        init_nn: Initial neural network after training.
        gen_nn: Generative neural network after training.
    """
    init_nn = LazyNet(widths)  # Initialize the neural network
    init_nn.reset_parameters()  # Reset parameters for the initial network

    # Generate linear data based on the provided coefficients
    X, Y = generate_linear_data(beta=beta, N=N, corr=corr)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=1)  # Split data into training and validation sets

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    gen_nn = LazyNet(widths)  # Initialize the generative neural network
    gen_nn.init_parameters(init_nn)  # Initialize parameters from the initial network

    # Create a version of the training data with the second feature set to zero
    X_train_drop = X_train.clone()
    X_train_drop[:, 1] = 0
    dm = FlexDataModule(X_train_drop, y_train, X_train_drop, y_train)  # Create a data module for training

    # Train the generative neural network
    trainer = pl.Trainer(max_epochs=10, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(gen_nn, dm)

    return init_nn, gen_nn  # Return the initial and generative networks

def gen_ntk_data(gen_nn, X, sd=5):
    """Generate data using the neural tangent kernel approach.

    Args:
        gen_nn: Generative neural network to use for generating outputs.
        X: Input features.
        sd: Standard deviation for the noise added to the output.

    Returns:
        Y: Generated output labels.
    """
    N = X.shape[0]  # Number of samples
    X_drop = X.clone()  # Create a copy of the input features
    X_drop[:, 1] = 0  # Set the second feature to zero

    # Generate output labels using the generative network and add noise
    Y = gen_nn(X_drop) + torch.normal(0, sd, size=(N, 1))

    return Y  # Return the generated output labels



def get_k(widths, X, n=100000):
    """Calculate the Neural Tangent Kernel (NTK) for a neural network with specified widths.

    Args:
        widths: List of layer widths for the neural network.
        X: Input data tensor.
        n: Number of samples for the NTK calculation.

    Returns:
        K: The computed Neural Tangent Kernel.
    """
    # Set the width of the hidden layers to n
    for i in range(len(widths)):
        if 0 < i < len(widths) - 1:
            widths[i] = n

    # Initialize the neural network
    full_nn = LazyNet(widths, lr=0.01)
    full_nn.reset_parameters()  # Reset parameters for the network

    tmp = full_nn(X)  # Forward pass through the network
    K = full_nn.NTK()  # Compute the Neural Tangent Kernel
    return K  # Return the computed NTK

def nn_mse_exp(X, sd, beta, widths, gen_nn, lr=0.01, max_iter=2000, plot=False):
    """Perform an experiment with a neural network and calculate the Mean Squared Error.

    Args:
        X: Input data tensor.
        sd: Standard deviation for noise in generated data.
        beta: Coefficients for the true model.
        widths: List of layer widths for the neural network.
        gen_nn: Generative neural network to produce target outputs.
        lr: Learning rate for training the neural networks.
        max_iter: Maximum number of iterations for training.
        plot: Boolean flag to determine whether to plot the training loss.

    Returns:
        Number of training samples, chosen T, and loss at T.
    """
    full_nn = LazyNet(widths, lr=lr)  # Initialize the full neural network
    full_nn.reset_parameters()  # Reset parameters for the network

    Y = gen_ntk_data(gen_nn, X, sd=sd)  # Generate target outputs using the generative network

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=1)
    X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert to tensor
    X_val = torch.tensor(X_val, dtype=torch.float32)  # Convert to tensor
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Convert to tensor
    y_val = torch.tensor(y_val, dtype=torch.float32)  # Convert to tensor
    dm = FlexDataModule(X_train, y_train, X_val, y_val)  # Create data module

    # Train the full network
    print('train full')
    early_stopping = EarlyStopping('val_loss', min_delta=1e-5)  # Early stopping callback
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=800)
    trainer.fit(full_nn, dm)  # Fit the model to the data
    print('full train done')

    # Create a version of the training data with the second feature set to zero
    X_train_drop = X_train.clone()
    X_train_drop[:, 1] = 0

    # Calculate the NTK for the modified input
    N = X_train.shape[0]  # Number of training samples
    k = get_k(widths, X_train_drop)  # Get the NTK for the dropped input

    # Generate residuals between the target output and the full network's output
    yR = gen_nn(X_train_drop) + (beta[1] * X_train_drop[:, 1])[:, None] - full_nn(X_train_drop)

    kinv = torch.linalg.pinv(k)  # Compute the pseudo-inverse of the NTK

    R2 = yR.T @ kinv @ yR  # Calculate R^2
    sigma = sd**2 + 1  # Calculate variance
    c = R2 * N / (4 * np.exp(1)**2 * sigma * lr**2)  # Compute constant for later calculations
    k = k / N  # Normalize the NTK

    T = np.arange(1, max_iter)  # Create an array for T values
    c = c.detach().numpy()  # Detach constant from the computation graph
    rhs = c / T**2  # Compute right-hand side for comparison

    # Calculate eigenvalues and rank of the NTK
    k = k.detach().numpy()
    e, s = np.linalg.eig(k)  # Eigenvalue decomposition
    r = np.linalg.matrix_rank(k)  # Calculate the rank of the NTK

    er = np.real(e[:r])  # Get real parts of the eigenvalues
    lhs = np.zeros((r, 2, len(T)))  # Initialize left-hand side for comparison
    lhs[:, 0, :] = er[:, None]  # First column is the eigenvalues
    lhs[:, 1, :] = 1 / lr / T  # Second column is scaled T values

    lhs = np.sum(np.min(lhs, axis=1), axis=0)  # Sum the minimum values along the specified axes

    # Find the first index where lhs exceeds rhs
    Tind = np.argmax(lhs > rhs)
    if Tind == 0:
        print(er)  # Print eigenvalues if no index is found
        print(lhs[:10])  # Print first 10 left-hand side values
        print(rhs[:10])  # Print first 10 right-hand side values
        print(R2)  # Print R^2 value
        print(c)  # Print constant

    T = T[Tind] - 1  # Choose the optimal T

    lazy_nn = LazyNet(widths, lr=lr)  # Initialize the lazy neural network
    lazy_nn.init_parameters(full_nn)  # Initialize parameters from the full network
    X_val_drop = X_val.clone()  # Clone validation set for modification
    X_val_drop[:, 1] = 0  # Set second feature to zero

    # Create a data module for training the lazy network
    dm = FlexDataModule(X_train_drop, y_train, X_train_drop, gen_nn(X_train_drop))
    cb = MetricTracker()  # Create a metric tracker for loss monitoring
    print(T)  # Print chosen T value

    # Determine how many iterations to run based on the plot flag
    runT = T + 300 if plot else T + 1
    runT = int(runT)
    print('lazy train')

    # Train the lazy network
    trainer = pl.Trainer(callbacks=[cb], max_epochs=runT, enable_progress_bar=True, enable_model_summary=False)
    trainer.fit(lazy_nn, dm)  # Fit the lazy neural network
    print('lazy done')
    print('N', X_train_drop.shape[0])  # Print the number of training samples

    print('chosen T', T)  # Print the chosen T value
    print(cb.loss[T])  # Print the loss at T

    # If plotting is enabled, create a plot of the loss over iterations
    if plot:
        plt.plot(np.arange(runT + 1), cb.loss)  # Plot the loss
        print('min', np.min(cb.loss))  # Print the minimum loss
        plt.grid()  # Add grid to the plot
        plt.show()  # Show the plot

    return X_train_drop.shape[0], T, cb.loss[T]  # Return the number of training samples, chosen T, and loss at T





def mse_exp(beta, X, sd, N, lr=0.01, max_iter=200):
    """Perform a mean squared error experiment with a neural network.

    Args:
        beta: Coefficients for generating output.
        X: Input features.
        sd: Standard deviation for the noise added to the output.
        N: Number of samples to generate.
        lr: Learning rate for the neural network.
        max_iter: Maximum iterations for training the network.

    Returns:
        Tuple containing the number of samples used, the optimal number of epochs (T), and the corresponding loss.
    """
    p = beta.shape[0]  # Number of features
    widths = [p, 50, 1]  # Define the architecture of the neural network
    full_nn = LazyNet(widths, lr=lr)  # Initialize the full neural network
    full_nn.reset_parameters()  # Reset the network parameters

    # Generate output data with added noise
    Y = X @ beta[:, None] + torch.normal(0, sd, size=(N, 1))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=1)
    X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert training features to tensor
    X_val = torch.tensor(X_val, dtype=torch.float32)  # Convert validation features to tensor
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Convert training output to tensor
    y_val = torch.tensor(y_val, dtype=torch.float32)  # Convert validation output to tensor
    dm = FlexDataModule(X_train, y_train, X_val, y_val)  # Create a data module for training

    # Train the full network
    early_stopping = EarlyStopping('val_loss', min_delta=1e-5)  # Define early stopping
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=800, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(full_nn, dm)  # Fit the full neural network

    # Cross-validation to find optimal number of epochs (T)
    n_k = 5  # Number of folds for cross-validation
    kf = KFold(n_splits=n_k, random_state=1, shuffle=True)  # Initialize K-Fold cross-validation
    X_train_drop = X_train.clone()  # Clone training data
    X_train_drop[:, 1] = 0  # Drop the second feature

    loss_all = np.zeros((n_k, max_iter + 1))  # Initialize array to store losses
    for i, (train_index, val_index) in enumerate(kf.split(X_train_drop)):
        # Split data into training and validation sets for this fold
        X_train_tmp = X_train_drop[train_index]
        y_train_tmp = y_train[train_index]
        X_val_tmp = X_train_drop[val_index]
        y_val_tmp = y_train[val_index]

        tmp_dm = FlexDataModule(X_train_tmp, y_train_tmp, X_val_tmp, y_val_tmp)  # Create data module for this fold

        tmp_nn = LazyNet(widths, lr=lr)  # Initialize a temporary neural network
        cb = MetricTracker()  # Initialize a metric tracker for monitoring loss
        tmp_nn.init_parameters(full_nn)  # Initialize the temporary network with parameters from the full network

        # Train the temporary network
        trainer = pl.Trainer(callbacks=[cb], max_epochs=max_iter, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(tmp_nn, tmp_dm)  # Fit the model to the data
        
        loss_all[i, :] = cb.loss  # Store the loss for this fold
        print('\n')  # New line for readability in output

    T = np.argmin(np.sum(loss_all, axis=0)[:])  # Determine the optimal number of epochs (T)

    # Train the lazy network
    lazy_nn = LazyNet(widths, lr=lr)  # Initialize the lazy neural network
    lazy_nn.init_parameters(full_nn)  # Initialize with parameters from the full network
    X_val_drop = X_val.clone()  # Clone validation data
    X_val_drop[:, 1] = 0  # Drop the second feature

    # Create a data module for lazy training
    dm = FlexDataModule(X_train_drop, y_train, X_train_drop, X_train_drop @ beta[:, None])
    cb = MetricTracker()  # Initialize a metric tracker for the lazy network

    # Train the lazy network
    trainer = pl.Trainer(callbacks=[cb], max_epochs=max_iter, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(lazy_nn, dm)  # Fit the lazy network to the data
    print('N', X_train_drop.shape[0])  # Print the number of samples used

    # Print and plot results
    print('cv T', T)  # Print the optimal number of epochs
    print(cb.loss[T])  # Print the loss at optimal T
    plt.plot(np.arange(max_iter + 1), cb.loss)  # Plot the loss over epochs
    print('min', np.min(cb.loss))  # Print minimum loss
    plt.grid()  # Add grid to plot
    plt.show()  # Show plot

    return X_train_drop.shape[0], T, cb.loss[T]  # Return number of samples, optimal T, and corresponding loss

def generate_logistic_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    """Generate logistic regression data.

    Args:
        beta: Coefficients for generating output.
        sigma: Standard deviation of the noise added to the output.
        N: Number of samples to generate.
        seed: Random seed for reproducibility.
        corr: Correlation coefficient for generating input data.

    Returns:
        Tuple containing the generated input data tensor and the output data tensor.
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    cov = [[1, corr], [corr, 1]]  # Create covariance matrix
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    p = beta.shape[0]  # Number of coefficients

    # Generate input data from normal distribution
    X = np.random.normal(0, 1, size=(N, p))
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)  # Correlated inputs
    normal_noise = np.random.normal(0, sigma, size=N)  # Generate noise

    # Compute expected output using the logistic model
    EY = np.exp(X @ beta) / (1 + np.exp(X @ beta))
    Y = EY + normal_noise  # Add noise to expected output

    X = torch.tensor(X, dtype=torch.float32)  # Convert inputs to tensor
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)  # Convert outputs to tensor and reshape

    return X, Y  # Return input and output tensors


def generate_linear_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    """Generate linear regression data with correlated features.

    Args:
        beta: Coefficients for generating output.
        sigma: Standard deviation of the noise added to the output.
        N: Number of samples to generate.
        seed: Random seed for reproducibility.
        corr: Correlation coefficient for generating input features.

    Returns:
        Tuple containing the generated input data tensor and the output data tensor.
    """
    random.seed(seed)  # Set the random seed for reproducibility
    cov = [[1, corr], [corr, 1]]  # Create covariance matrix for correlated features
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    p = beta.shape[0]  # Number of coefficients

    # Calculate true variance for each coefficient, adjusting for correlation
    VI_true = beta ** 2  
    VI_true[0:2] = VI_true[0:2] * (1 - corr ** 2)

    # Generate input data
    X = np.random.normal(0, 1, size=(N, p))  # Generate normally distributed features
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)  # Introduce correlated features

    normal_noise = np.random.normal(0, sigma, size=N)  # Generate noise to add to output

    # Calculate expected output
    EY = np.matmul(X, beta)  # Linear combination of inputs and coefficients
    Y = EY + normal_noise  # Add noise to expected output

    # Convert inputs and outputs to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)  # Reshape output tensor to be a column vector

    return X, Y  # Return input and output tensors



   # Flexible data modules for more complicated/principled data handling
class FlexDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, seed=711, batch_size: int = 32):
        """Initialize the data module.

        Args:
            X_train: Training feature data.
            y_train: Training target data.
            X_val: Validation feature data.
            y_val: Validation target data.
            seed: Random seed for reproducibility.
            batch_size: Batch size for data loaders.
        """
        super().__init__()
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the datasets for training and validation."""
        self.train = torch.utils.data.TensorDataset(
            self.X_train.clone().detach(),
            self.y_train.clone().detach()
        )
        self.val = torch.utils.data.TensorDataset(
            self.X_val.clone().detach(),
            self.y_val.clone().detach()
        )

    def train_dataloader(self):
        """Return the training data loader."""
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return the validation data loader."""
        tmp_bs = len(self.val)  # Set batch size to the size of the validation set
        return DataLoader(self.val, batch_size=tmp_bs)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size)


class MetricTracker(Callback):
    """Tracks validation loss during training."""
    
    def __init__(self):
        self.loss = []  # List to store loss values

    def on_validation_batch_end(self, trainer, LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each validation batch.

        Args:
            trainer: The trainer instance.
            LightningModule: The Lightning module being trained.
            outputs: Outputs from the model.
            batch: The current batch of data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        mse = outputs['val_loss']  # Access validation loss from outputs
        self.loss.append(mse.item())  # Append loss to the loss list

    def on_validation_epoch_end(self, trainer, LightningModule):
        """Called at the end of each validation epoch."""
        pass  # No additional actions needed at this point
