import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from scipy import stats as st
import time
from IPython.utils import io
from sklearn.ensemble import RandomForestRegressor
import tqdm
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


"""
VI helpers
"""

def calculate_cvg(est, se, true_vi, level=.05):
    """
    Calculate coverage of a confidence interval.
    
    :param est: estimated value
    :param se: standard error
    :param true_vi: true value to check coverage
    :param level: confidence level (default is 0.05 for 95% CI)
    :return: 1 if true_vi is within the CI, otherwise 0
    """
    z = st.norm.ppf((1 - level / 2))  # Z-score for the given confidence level
    lb = est - z * se  # Lower bound of CI
    ub = est + z * se  # Upper bound of CI
    return 1 if true_vi >= lb and true_vi <= ub else 0  # Return coverage indicator

def dropout(X, grp):
    """
    Perform dropout by replacing specified features with their mean.
    
    :param X: input data
    :param grp: index or list of indices of features to dropout
    :return: modified input data with dropped features replaced by their mean
    """
    X = np.array(X)  # Convert input to a NumPy array
    N = X.shape[0]  # Number of samples
    X_change = np.copy(X)  # Copy of the original data

    # Replace specified features with their mean
    if isinstance(grp, (np.int64, int)):
        X_change[:, grp] = np.ones(N) * np.mean(X[:, grp])
    else:
        for j in grp:
            X_change[:, j] = np.ones(N) * np.mean(X[:, j])

    return torch.tensor(X_change, dtype=torch.float32)  # Return modified data as a tensor

def retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=1e-3):
    """
    Retrain the neural network on modified training data.
    
    :param p: input dimension
    :param hidden_layers: list of hidden layer sizes
    :param j: index for the retraining process
    :param X_train_change: modified training features
    :param y_train: training targets
    :param X_test_change: modified test features
    :param tol: tolerance for early stopping (default is 1e-3)
    :return: predictions on modified training and test data
    """
    retrain_nn = NN4vi(p, hidden_layers, 1)  # Initialize a new instance of the neural network
    early_stopping = EarlyStopping('val_loss', min_delta=tol)  # Early stopping callback

    # Create TensorDataset and DataLoader for training
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_change, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(trainset, batch_size=256)  # DataLoader for batching

    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=100)  # Trainer for training
    with io.capture_output() as captured:
        trainer.fit(retrain_nn, train_loader, train_loader)  # Train the model

    # Get predictions for training and test datasets
    retrain_pred_train = retrain_nn(X_train_change)
    retrain_pred_test = retrain_nn(X_test_change)
    return retrain_pred_train, retrain_pred_test  # Return predictions

def fake_retrain(p, full_nn, hidden_layers, j, X_train_change, y_train, X_test_change, tol=1e-5, max_epochs=10):
    """
    Retrain the neural network using the state of a previously trained model.
    
    :param p: input dimension
    :param full_nn: previously trained neural network model
    :param hidden_layers: list of hidden layer sizes
    :param j: index for the retraining process
    :param X_train_change: modified training features
    :param y_train: training targets
    :param X_test_change: modified test features
    :param tol: tolerance for early stopping (default is 1e-5)
    :param max_epochs: maximum epochs for retraining (default is 10)
    :return: predictions on modified training and test data
    """
    retrain_nn = NN4vi(p, hidden_layers, 1)  # Initialize a new instance of the neural network
    retrain_nn.load_state_dict(full_nn.state_dict())  # Load weights from the trained model
    early_stopping = EarlyStopping('val_loss', min_delta=tol)  # Early stopping callback

    # Create TensorDataset and DataLoader for training
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_change, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(trainset, batch_size=256)  # DataLoader for batching

    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=max_epochs)  # Trainer for training
    with io.capture_output() as captured:
        trainer.fit(retrain_nn, train_loader, train_loader)  # Train the model

    # Get predictions for training and test datasets
    retrain_pred_train = retrain_nn(X_train_change)
    retrain_pred_test = retrain_nn(X_test_change)
    return retrain_pred_train, retrain_pred_test  # Return predictions

"""
Lazy training
"""

def flat_tensors(T_list: list):
    """
    Flatten a list of tensors into a single vector and store their original shapes for recovery.
    
    :param T_list: list of tensors to flatten
    :return: Tuple containing the flattened tensor and the list of original shapes
    """
    info = [t.shape for t in T_list]  # Store the shapes of the original tensors
    res = torch.cat([t.reshape(-1) for t in T_list])  # Concatenate tensors into a single vector
    return res, info  # Return flattened tensor and original shapes

def recover_tensors(T: torch.Tensor, info: list):
    """
    Recover parameter tensors from a flattened tensor using stored shapes.
    
    :param T: flattened tensor
    :param info: list of original shapes of the tensors
    :return: list of recovered tensors
    """
    i = 0
    res = []  # List to store recovered tensors
    for s in info:
        len_s = np.prod(s)  # Calculate the total number of elements in the tensor
        res.append(T[i:i + len_s].reshape(s))  # Reshape and append the tensor to the result list
        i += len_s  # Move the index forward
    return res  # Return the list of recovered tensors

def extract_grad(X, full_nn):
    """
    Extract gradients from the trained network for the given input.
    
    :param X: input data
    :param full_nn: trained neural network
    :return: tuple containing gradients, flattened parameters, and shape information
    """
    grads = []  # List to store gradients
    n = X.shape[0]  # Number of samples
    params_full = tuple(full_nn.parameters())  # Get model parameters
    flat_params, shape_info = flat_tensors(params_full)  # Flatten parameters and store shapes
    for i in range(n):
        # Calculate the first-order gradient with respect to all parameters
        if len(X.shape) > 2:
            yi = full_nn(X[[i]])  # Forward pass for batched input
        else:
            yi = full_nn(X[i])  # Forward pass for single input
        this_grad = torch.autograd.grad(yi, params_full, create_graph=True)  # Compute gradients
        flat_this_grad, _ = flat_tensors(this_grad)  # Flatten gradients
        grads.append(flat_this_grad)  # Append to gradients list
    grads = np.array([grad.detach().numpy() for grad in grads])  # Convert gradients to a NumPy array
    return grads, flat_params, shape_info  # Return gradients, flattened parameters, and shape info

def lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info, X_train, y_train, X_test, lam):
    """
    Perform lazy prediction using the extracted gradients and parameters.
    
    :param grads: extracted gradients
    :param flat_params: flattened parameters of the trained network
    :param full_nn: trained neural network
    :param hidden_layers: list of hidden layer sizes
    :param shape_info: original shapes of the parameters
    :param X_train: training input data
    :param y_train: training targets
    :param X_test: test input data
    :param lam: regularization parameter for Ridge regression
    :return: predictions for training and test data
    """
    _, p = X_train.shape  # Get the number of features
    dr_pred_train = full_nn(X_train)  # Get predictions from the trained model
    lazy = Ridge(alpha=lam).fit(grads, y_train - dr_pred_train.detach().numpy())  # Fit Ridge regression
    delta = lazy.coef_  # Get the coefficients from the Ridge model
    lazy_retrain_params = torch.FloatTensor(delta) + flat_params  # Adjust parameters with delta
    lazy_retrain_Tlist = recover_tensors(lazy_retrain_params.reshape(-1), shape_info)  # Recover tensors
    lazy_retrain_nn = NN4vi(p, hidden_layers, 1)  # Initialize a new instance of the neural network

    # Load adjusted parameters into the new model
    for k, param in enumerate(lazy_retrain_nn.parameters()):
        param.data = lazy_retrain_Tlist[k]

    # Get predictions for training and test datasets
    lazy_pred_train = lazy_retrain_nn(X_train)
    lazy_pred_test = lazy_retrain_nn(X_test)
    return lazy_pred_train, lazy_pred_test  # Return predictions

def lazy_train_cv(full_nn, X_train_change, X_test_change, y_train, hidden_layers,
                  lam_path=np.logspace(-3, 3, 20), file=False):
    """
    Perform lazy training with cross-validation to select the best regularization parameter.
    
    :param full_nn: trained neural network
    :param X_train_change: modified training features
    :param X_test_change: modified test features
    :param y_train: training targets
    :param hidden_layers: list of hidden layer sizes
    :param lam_path: range of regularization parameters to test
    :param file: optional parameter for file saving (not used here)
    :return: predictions for training and test data, and error dataframe
    """
    kf = KFold(n_splits=3, shuffle=True)  # Initialize KFold cross-validation
    errors = []  # List to store errors for each fold and lambda

    # Extract gradients and parameters from the full network
    grads, flat_params, shape_info = extract_grad(X_train_change, full_nn)

    # Loop over each regularization parameter
    for lam in lam_path:
        for train, test in kf.split(X_train_change):
            dr_pred_train = full_nn(X_train_change[train])  # Get predictions for training split
            grads_train = grads[train]  # Select gradients for the training split
            # Perform lazy prediction
            lazy_pred_train, lazy_pred_test = lazy_predict(grads_train, flat_params, full_nn, hidden_layers, shape_info,
                                                           X_train_change[train], y_train[train], X_train_change[test],
                                                           lam)
            # Compute and store MSE for the lazy predictions
            errors.append([lam, nn.MSELoss()(lazy_pred_test, y_train[test]).item()])
    
    # Create a DataFrame to summarize the errors
    errors = pd.DataFrame(errors, columns=['lam', 'mse'])
    # Select the lambda that minimizes the mean squared error
    lam = errors.groupby(['lam']).mse.mean().sort_values().index[0]
    print(lam)  # Output the selected lambda

    # Perform final lazy prediction using the best lambda
    lazy_pred_train, lazy_pred_test = lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info,
                                                   X_train_change, y_train, X_test_change, lam)
    return lazy_pred_train, lazy_pred_test, errors  # Return predictions and error DataFrame



"""
Experiment wrapped for faster simulations
"""

def vi_experiment_wrapper(X, y, network_width, ix=None, exp_iter=1, lambda_path=np.logspace(0, 2, 10),
                          lam='cv', lazy_init='train', do_retrain=True, include_linear=False, include_rf=False,
                          early_stop=False, max_epochs=100):
    """
    Wrapper function for conducting experiments with various model training strategies.

    :param X: numpy array, input features for training
    :param y: numpy array, target values for training
    :param network_width: int, width of the neural network
    :param ix: list or None, indices of features to experiment on (default is None, which uses all features)
    :param exp_iter: int, iteration number for random seed (used in train/test split)
    :param lambda_path: numpy array, range of regularization parameters for lazy training
    :param lam: str or float, 'cv' for cross-validation or a specific lambda value for lazy learning
    :param lazy_init: str, method for lazy initialization ('train' or 'random')
    :param do_retrain: bool, whether to perform retraining or not
    :param include_linear: bool, whether to include a linear regression model in the comparison
    :param include_rf: bool, whether to include a random forest model in the comparison
    :param early_stop: bool, whether to apply early stopping during retraining
    :param max_epochs: int, maximum number of epochs for retraining
    :return: DataFrame containing results of the experiment
    """
    n, p = X.shape  # Get the number of samples (n) and number of features (p)
    
    # If no specific indices are provided, use all feature indices
    if ix is None:
        ix = np.arange(p)
    
    hidden_layers = [network_width]  # Define the structure of hidden layers
    tol = 1e-3  # Tolerance for early stopping
    results = []  # Initialize results list
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=exp_iter)
    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1, 1))  # Create Tensor dataset
    train_loader = DataLoader(trainset, batch_size=256)  # Create DataLoader for training
    
    # Initialize and train the full neural network model
    full_nn = NN4vi(p, hidden_layers, 1)
    early_stopping = EarlyStopping('val_loss', min_delta=tol)  # Early stopping callback
    trainer = pl.Trainer(callbacks=[early_stopping])  # Initialize trainer
    t0 = time.time()  # Start timer
    with io.capture_output() as captured: 
        trainer.fit(full_nn, train_loader, train_loader)  # Train the model
    full_time = time.time() - t0  # Calculate training time
    full_pred_test = full_nn(X_test)  # Predict on the test set
    
    # Append results for the full model
    results.append(['all', 'full model', full_time, 0,
                    nn.MSELoss()(full_nn(X_train), y_train).item(),
                    nn.MSELoss()(full_pred_test, y_test).item()])

    # Include linear regression model if specified
    if include_linear:
        lm = LinearRegression()
        lm.fit(X_train.detach().numpy(), y_train.detach().numpy())  # Train linear regression model

    # Include random forest model if specified
    if include_rf:
        rf = RandomForestRegressor()
        rf.fit(X_train.detach().numpy(), y_train.detach().numpy())  # Train random forest model

    # Loop through each feature index specified for dropout experiments
    for j in ix:
        varr = 'X' + str(j + 1)  # Create a label for the current variable
        
        # DROPOUT: Change the current feature to its mean and evaluate
        X_test_change = dropout(X_test, j)  # Apply dropout to test set
        X_train_change = dropout(X_train, j)  # Apply dropout to training set
        dr_pred_train = full_nn(X_train_change)  # Predict on modified training set
        dr_pred_test = full_nn(X_test_change)    # Predict on modified test set
        dr_train_loss = nn.MSELoss()(dr_pred_train, y_train).item()  # Calculate training loss
        dr_test_loss = nn.MSELoss()(dr_pred_test, y_test).item()  # Calculate test loss
        dr_vi = nn.MSELoss()(dr_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()  # Calculate variance impact

        # Calculate variance estimate
        eps_j = ((y_test - dr_pred_test) ** 2).detach().numpy().reshape(1, -1)
        eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
        se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])  # Standard error calculation

        results.append([varr, 'dropout', 0, dr_vi, dr_train_loss, dr_test_loss, se])  # Append dropout results

        # LAZY: Use lazy training method
        t0 = time.time()  # Start timer

        if lam == 'cv':
            # Perform lazy training with cross-validation
            lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(full_nn, X_train_change, X_test_change, y_train,
                                                                     hidden_layers, lam_path=lambda_path)
        else:
            # Extract gradients and perform lazy prediction
            grads, flat_params, shape_info = extract_grad(X_train_change, full_nn)
            lazy_pred_train, lazy_pred_test = lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info,
                                                           X_train_change, y_train, X_test_change, lam)
        lazy_time = time.time() - t0  # Calculate lazy training time
        lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()  # Calculate lazy training loss
        lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()  # Calculate lazy test loss
        lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()  # Variance impact for lazy prediction

        # Calculate variance estimate for lazy prediction
        eps_j = ((y_test - lazy_pred_test) ** 2).detach().numpy().reshape(1, -1)
        eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
        se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])  # Standard error calculation

        results.append([varr, 'lazy', lazy_time, lazy_vi, lazy_train_loss, lazy_test_loss, se])  # Append lazy results

        # LAZY with random initialization
        if lazy_init == 'random':
            t0 = time.time()  # Start timer
            random_nn = NN4vi(p, hidden_layers, 1)  # Initialize a new neural network
            params_full = tuple(full_nn.parameters())  # Get parameters from the full model
            flat_params, shape_info = flat_tensors(params_full)  # Flatten parameters
            lazy_retrain_Tlist = recover_tensors(flat_params.reshape(-1), shape_info)  # Recover tensors from flattened params
            
            # Add noise to parameters for random initialization
            for k, param in enumerate(random_nn.parameters()):
                param.data = lazy_retrain_Tlist[k] + np.random.normal(size=lazy_retrain_Tlist[k].shape)

            # Train the random neural network with lazy training
            lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(random_nn, X_train_change, X_test_change, y_train,
                                                                     hidden_layers, lam_path=lambda_path)
            lazy_time = time.time() - t0  # Calculate lazy training time
            lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()  # Calculate lazy training loss
            lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()  # Calculate lazy test loss
            lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()  # Variance impact for lazy random

            # Append random lazy results
            results.append([varr, 'lazy_random', lazy_time, lazy_vi, lazy_train_loss, lazy_test_loss])

        # RETRAIN: Retrain the model with modified input
        if do_retrain:
            t0 = time.time()  # Start timer
            retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=tol)
            retrain_time = time.time() - t0  # Calculate retraining time
            vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_pred_test).item()  # Variance impact for retrain
            loss_rt_test = nn.MSELoss()(retrain_pred_test, y_test).item()  # Test loss after retraining
            loss_rt_train = nn.MSELoss()(retrain_pred_train, y_train).item()  # Training loss after retraining

            # Calculate variance estimate for retraining
            eps_j = ((y_test - retrain_pred_test) ** 2).detach().numpy().reshape(1, -1)
            eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
            se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])  # Standard error calculation

            results.append([varr, 'retrain', retrain_time, vi_retrain, loss_rt_train, loss_rt_test, se])  # Append retrain results

        # Early stopping retraining
        if early_stop:
            t0 = time.time()  # Start timer
            retrain_pred_train, retrain_pred_test = fake_retrain(p, full_nn, hidden_layers, j, X_train_change, y_train,
                                                                 X_test_change, tol=tol, max_epochs=max_epochs)
            retrain_time = time.time() - t0  # Calculate retraining time
            vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_pred_test).item()  # Variance impact for early stopping
            loss_rt_test = nn.MSELoss()(retrain_pred_test, y_test).item()  # Test loss after early stopping
            loss_rt_train = nn.MSELoss()(retrain_pred_train, y_train).item()  # Training loss after early stopping

            # Calculate variance estimate for early stopping retraining
            eps_j = ((y_test - retrain_pred_test) ** 2).detach().numpy().reshape(1, -1)
            eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
            se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])  # Standard error calculation

            results.append([varr, 'early_stopping', retrain_time, vi_retrain, loss_rt_train, loss_rt_test, se])  # Append early stopping results

        # LINEAR RETRAIN: Evaluate with linear regression after dropout
        if include_linear:
            t0 = time.time()  # Start timer
            lmj = LinearRegression()  # Initialize linear regression model
            lmj.fit(X_train_change.detach().numpy(), y_train.detach().numpy())  # Train linear regression model
            lin_time = time.time() - t0  # Calculate linear model training time
            vi_linear = mse(lmj.predict(X_test_change.detach().numpy()), y_test.detach().numpy()) - mse(lm.predict(X_test.detach().numpy()), y_test.detach().numpy())  # Variance impact for linear regression
            loss_rt_test = mse(lmj.predict(X_test_change.detach().numpy()), y_test.detach().numpy())  # Test loss for linear model
            loss_rt_train = mse(lmj.predict(X_train_change.detach().numpy()), y_train.detach().numpy())  # Training loss for linear model
            results.append([varr, 'ols', lin_time, vi_linear, loss_rt_train, loss_rt_test])  # Append linear results

        # RANDOM FOREST RETRAIN: Evaluate with random forest after dropout
        if include_rf:
            t0 = time.time()  # Start timer
            rfj = RandomForestRegressor()  # Initialize random forest model
            rfj.fit(X_train_change.detach().numpy(), y_train.detach().numpy())  # Train random forest model
            lin_time = time.time() - t0  # Calculate random forest training time
            vi_linear = mse(rfj.predict(X_test_change.detach().numpy()), y_test.detach().numpy()) - mse(rf.predict(X_test.detach().numpy()), y_test.detach().numpy())  # Variance impact for random forest
            loss_rt_test = mse(rfj.predict(X_test_change.detach().numpy()), y_test.detach().numpy())  # Test loss for random forest
            loss_rt_train = mse(rfj.predict(X_train_change.detach().numpy()), y_train.detach().numpy())  # Training loss for random forest
            results.append([varr, 'rf', lin_time, vi_linear, loss_rt_train, loss_rt_test])  # Append random forest results

    # Create a DataFrame from the results and return it
    df = pd.DataFrame(results, columns=['variable', 'method', 'time', 'vi', 'train_loss', 'test_loss', 'se'])
    return df  # Return the DataFrame with experiment results


def boxplot_2d(x, y, ax, co='g', whis=1.5, method=''):
    """
    Draw a 2D boxplot on the given axes.
    
    :param x: array-like, data for the x-axis
    :param y: array-like, data for the y-axis
    :param ax: matplotlib axes object to draw the boxplot on
    :param co: color for the box and lines
    :param whis: defines the reach of the whiskers (default is 1.5)
    :param method: method name to label the central point
    """

    # Calculate the x and y quartiles
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]  # Q1, median, Q3
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]  # Q1, median, Q3

    ## Create the box
    box = Rectangle(
        (xlimits[0], ylimits[0]),  # bottom-left corner
        (xlimits[2] - xlimits[0]),  # width of the box
        (ylimits[2] - ylimits[0]),  # height of the box
        ec=co,  # edge color
        color=co,  # box color
        zorder=0  # drawing order
    )
    ax.add_patch(box)  # Add the box to the axes

    ## Draw the x median line
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],  # x position is the median, y spans the box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(vline)  # Add the vertical median line to the axes

    ## Draw the y median line
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],  # y position is the median, x spans the box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(hline)  # Add the horizontal median line to the axes

    ## Plot the central point (median point)
    ax.plot([xlimits[1]], [ylimits[1]], color=co, marker='o', label=method)  # Central point

    ## Calculate the interquartile range (IQR)
    iqr = xlimits[2] - xlimits[0]

    ## Left whisker
    left = np.min(x[x > xlimits[0] - whis * iqr])  # Calculate the left whisker
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1], ylimits[1]],  # Draw horizontal line from whisker to box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_line)  # Add the left whisker line to the axes
    whisker_bar = Line2D(
        [left, left], [ylimits[0], ylimits[2]],  # Draw vertical line at left whisker position
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_bar)  # Add the left whisker bar to the axes

    ## Right whisker
    right = np.max(x[x < xlimits[2] + whis * iqr])  # Calculate the right whisker
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]],  # Draw horizontal line from whisker to box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_line)  # Add the right whisker line to the axes
    whisker_bar = Line2D(
        [right, right], [ylimits[0], ylimits[2]],  # Draw vertical line at right whisker position
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_bar)  # Add the right whisker bar to the axes

    ## Calculate the y interquartile range (IQR)
    iqr = ylimits[2] - ylimits[0]

    ## Bottom whisker
    bottom = np.min(y[y > ylimits[0] - whis * iqr])  # Calculate the bottom whisker
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [bottom, ylimits[0]],  # Draw vertical line from whisker to box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_line)  # Add the bottom whisker line to the axes
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [bottom, bottom],  # Draw horizontal line at bottom whisker position
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_bar)  # Add the bottom whisker bar to the axes

    ## Top whisker
    top = np.max(y[y < ylimits[2] + whis * iqr])  # Calculate the top whisker
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [top, ylimits[2]],  # Draw vertical line from whisker to box
        color=co,  # line color
        zorder=1  # drawing order
    )
    ax.add_line(whisker_line)  # Add the top whisker line to the axes
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [top, top],  # Draw horizontal line at top whisker position
        color=co,  # line color
        zorder=1  # drawing order
    )
