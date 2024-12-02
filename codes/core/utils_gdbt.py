import numpy as np
from catboost import Pool, CatBoostRegressor
import time
import random
from itertools import chain, combinations
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from itertools import product, permutations, combinations
from math import factorial
from sympy import prime

def powerset(iterable):
    """Generate the powerset of the given iterable.

    Args:
        iterable: An iterable (e.g., list, set) from which to generate the powerset.

    Returns:
        A generator that yields all combinations of the iterable's elements.
    """
    s = list(iterable)  # Convert the iterable to a list
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))  # Generate all combinations

def getprime(k):
    """Get the first k prime numbers, including 1 as the first element.

    Args:
        k: The number of prime numbers to return.

    Returns:
        A list containing 1 and the first k prime numbers.
    """
    res = [1]  # Initialize the result list with 1
    for i in range(1, k+1):
        res.append(prime(i))  # Append the ith prime number to the result list
    return res


def cal_stationk(data, model):
    """Calculate the kernel matrix K for a given dataset and CatBoost model.

    Args:
        data: A numpy array containing the input features.
        model: A trained CatBoost model.

    Returns:
        A numpy array representing the kernel matrix K.
    """
    # Get model parameters
    n_depth = model.get_param('depth')  # Depth of the trees in the model
    borders = model.get_borders()  # Retrieve the feature borders for splitting

    feature_used = []  # List to store features used for splits
    partitions = []  # List to store number of partitions for each feature
    for key, value in borders.items():
        if len(value):
            feature_used.append(key)  # Add the feature if it has borders
            partitions.append(len(value))  # Count the number of partitions for the feature

    n_feature = len(feature_used)  # Total number of features used
    total_borders = np.sum(partitions)  # Total number of borders across all features
    cumpar = np.cumsum(partitions)  # Cumulative sum of partitions for indexing
    N = data.shape[0]  # Number of samples in the dataset

    K = np.zeros((N, N))  # Initialize the kernel matrix K with zeros

    # Iterate over all possible permutations of borders for tree depth
    for ele in permutations(range(total_borders), n_depth):
        tmp_tree = []  # List to store current tree structure
        for i in range(n_depth):
            for j in range(n_feature):
                if ele[i] < cumpar[j]:  # Determine the feature and border to use
                    tmp_id = j
                    feature_id = feature_used[tmp_id]  # Get the feature used for this split
                    border_id = ele[i] if j == 0 else ele[i] - cumpar[j - 1]  # Determine the border index
                    tmp_split = str(feature_id) + ',' + str(borders[feature_id][border_id])  # Create a string for the split
                    tmp_tree.append(tmp_split)  # Append the split to the tree
                    break

        # Compute leaf counts for the current tree structure
        sample_space = list(product([0, 1], repeat=n_depth) )  # All possible combinations of split directions
        primes = getprime(2 ** n_depth)  # Get prime numbers corresponding to leaf counts
        res = []  # To store the counts of samples reaching each leaf
        leafs = np.zeros(N, dtype=int)  # To track which leaf each sample belongs to

        # Iterate through each combination of split directions
        for i in range(len(sample_space)):
            inds = np.ones(N)  # Initialize indicator array for samples
            tmp_s = sample_space[i]  # Current split direction combination
            for j in range(n_depth):
                tmp_f, tmp_border = tmp_tree[j].split(',')  # Get the feature and border for the current split
                tmp_f = eval(tmp_f)  # Evaluate the feature index
                tmp_border = eval(tmp_border)  # Evaluate the border value
                tmp_ind = data[:, tmp_f] > tmp_border if tmp_s[j] else data[:, tmp_f] <= tmp_border  # Determine which samples satisfy the split condition
                inds = np.logical_and(inds, tmp_ind)  # Update indicator array

            Ni = np.sum(inds)  # Count of samples in the current leaf
            res.append(Ni)  # Store the count
            
            leafs[inds] = primes[i]  # Assign prime number to samples in this leaf

        leafs = leafs[:, None]  # Reshape leafs for matrix multiplication
        tmp_k = leafs @ leafs.T  # Compute outer product of leaf assignments
        for i in range(2 ** n_depth):
            if res[i] > 0:  # Only consider non-empty leaves
                tmp_ind = tmp_k == primes[i] ** 2  # Find indices corresponding to the current prime number
                K[tmp_ind] += N / res[i]  # Update kernel matrix based on the count of samples

    # Normalize the kernel matrix K by the number of samples and permutations
    permu_count = factorial(total_borders) / factorial(total_borders - n_depth)  # Number of permutations
    K = K / N / permu_count  # Final normalization

    return K  # Return the calculated kernel matrix


        
def generate_logistic_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    """Generate synthetic logistic regression data.

    Args:
        beta: Coefficients for the logistic model.
        sigma: Standard deviation of the noise to be added to the response variable.
        N: Number of samples to generate.
        seed: Seed for random number generation to ensure reproducibility.
        corr: Correlation coefficient for the first two features.

    Returns:
        Tuple of numpy arrays (X, Y) where:
        - X is the feature matrix of shape (N, p).
        - Y is the response variable vector of shape (N,).
    """
    random.seed(seed)  # Set the random seed for reproducibility
    cov = [[1, corr], [corr, 1]]  # Define the covariance matrix for correlated features
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    p = beta.shape[0]  # Number of features (length of beta)

    # Generate feature matrix X with normal distribution
    X = np.random.normal(0, 1, size=(N, p))

    # Replace the first two features with correlated random variables
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)

    # Generate normal noise
    normal_noise = np.random.normal(0, sigma, size=N)

    # Calculate the expected value using the logistic function
    EY = 1 / (1 + np.exp(-X @ beta))  # Logistic transformation of the linear combination

    # Generate the response variable Y by adding noise to the expected value
    Y = EY + normal_noise

    return X, Y  # Return the generated features and response variable





def cal_epk(data, model):
    """Calculate the empirical kernel for a given data set and CatBoost model.

    Args:
        data: Input feature matrix of shape (N, p) where N is the number of samples.
        model: Trained CatBoost model used to calculate leaf indices.

    Returns:
        A numpy array representing the empirical kernel matrix of shape (N, N).
    """
    n_iter = model.tree_count_  # Get the total number of trees in the model
    alloc = model.calc_leaf_indexes(data, n_iter - 1, n_iter).T[0, :]  # Get leaf allocations for each sample
    N = alloc.shape[0]  # Number of samples
    unique, counts = np.unique(alloc, return_counts=True)  # Unique leaf indices and their counts
    weights = N / counts  # Calculate weights for each leaf based on the number of samples

    out = np.zeros((N, N))  # Initialize the output kernel matrix
    cumcounts = np.cumsum(counts)  # Cumulative counts of samples in each leaf
    k = counts.shape[0]  # Number of unique leaves

    for i in range(k):
        if i == 0:
            out[0:cumcounts[i], 0:cumcounts[i]] = weights[i] * np.ones((counts[i], counts[i]))
        else:
            out[cumcounts[i-1]:cumcounts[i], cumcounts[i-1]:cumcounts[i]] = weights[i] * np.ones((counts[i], counts[i]))

    return out / N  # Return the empirical kernel normalized by the number of samples


def dropdata(X_train, X_val, X_test, dropi):
    """Drop a feature from the training, validation, and test sets by replacing it with the mean.

    Args:
        X_train: Training feature matrix of shape (N_train, p).
        X_val: Validation feature matrix of shape (N_val, p).
        X_test: Test feature matrix of shape (N_test, p).
        dropi: Index of the feature to be dropped (replaced with the mean).

    Returns:
        Tuple of modified feature matrices (X_train_drop, X_val_drop, X_test_drop).
    """
    X_train_drop = X_train.copy()  # Create a copy of the training data
    X_val_drop = X_val.copy()  # Create a copy of the validation data
    X_test_drop = X_test.copy()  # Create a copy of the test data

    # Replace the specified feature in each dataset with its mean
    X_train_drop[:, dropi] = np.mean(X_train_drop[:, dropi], axis=0)
    X_val_drop[:, dropi] = np.mean(X_val_drop[:, dropi], axis=0)
    X_test_drop[:, dropi] = np.mean(X_test_drop[:, dropi], axis=0)

    return X_train_drop, X_val_drop, X_test_drop  # Return the modified datasets


def generate_linear_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    """Generate synthetic linear regression data.

    Args:
        beta: Coefficients for the linear model.
        sigma: Standard deviation of the noise to be added to the response variable.
        N: Number of samples to generate.
        seed: Seed for random number generation to ensure reproducibility.
        corr: Correlation coefficient for the first two features.

    Returns:
        Tuple of numpy arrays (X, Y) where:
        - X is the feature matrix of shape (N, p).
        - Y is the response variable vector of shape (N,).
    """
    random.seed(seed)  # Set the random seed for reproducibility
    cov = [[1, corr], [corr, 1]]  # Define the covariance matrix for correlated features
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    p = beta.shape[0]  # Number of features (length of beta)
    VI_true = beta ** 2  # Calculate the variance of each feature

    # Adjust the variances of the first two features based on correlation
    VI_true[0:2] = VI_true[0:2] * (1 - corr ** 2)

    # Generate feature matrix X with normal distribution
    X = np.random.normal(0, 1, size=(N, p))

    # Replace the first two features with correlated random variables
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)

    # Generate normal noise
    normal_noise = np.random.normal(0, sigma, size=N)

    # Calculate the expected value as a linear combination of the features
    EY = np.matmul(X, beta)

    # Generate the response variable Y by adding noise to the expected value
    Y = EY + normal_noise

    return X, Y  # Return the generated features and response variable


def generate_simple_data(beta = [2,3,5],sigma = 1, N = 1000,seed = 1, k = 5,a = 1,c1 = 2,c2 = 4):

    rng = np.random.RandomState(seed= seed)
    dim = len(beta)
    beta = np.array(beta, dtype=float)
    X = np.zeros((N,dim))
    for i in range(dim):
        X[:,i] = rng.randint(a*i, a*i+k,size= N) * 2

    

   
    normal_noise = np.random.normal(0, sigma, size=N)
    
    EY = np.matmul(X, beta)
    Y = EY + normal_noise
    return X,Y, EY


def generate_indep_data(beta=[2, 3, 5], sigma=1, N=1000, seed=1, k=5, a=1, c1=2, c2=4):
    """Generate independent linear data.

    Args:
        beta: Coefficients for the linear model.
        sigma: Standard deviation of the noise to be added to the response variable.
        N: Number of samples to generate.
        seed: Seed for random number generation to ensure reproducibility.
        k: Range of random integers for generating features.
        a: Scaling factor for feature generation.
        c1, c2: Coefficients for potential feature interactions (not currently used).

    Returns:
        Tuple of numpy arrays (X, Y, EY) where:
        - X is the feature matrix of shape (N, dim).
        - Y is the response variable vector of shape (N,).
        - EY is the expected value (without noise).
    """
    dim = len(beta)  # Get the number of features
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    X = np.zeros((N, dim))  # Initialize feature matrix

    # Generate independent features based on the specified range and scaling
    for i in range(dim):
        X[:, i] = np.random.randint(a * i, a * i + k, size=N) * 2

    normal_noise = np.random.normal(0, sigma, size=N)  # Generate normal noise
    
    EY = np.matmul(X, beta)  # Calculate the expected values
    Y = EY + normal_noise  # Generate the response variable by adding noise
    return X, Y, EY  # Return the feature matrix, response variable, and expected values


def generate_nonlinear_data(beta=[2, 3, 5], b=3, sigma=1, N=1000, seed=1, k=5, a=1, c1=2, c2=4):
    """Generate nonlinear data based on a polynomial relationship.

    Args:
        beta: Coefficients for the linear model (the first element is the bias).
        b: Constant to be added to the expected value.
        sigma: Standard deviation of the noise to be added to the response variable.
        N: Number of samples to generate.
        seed: Seed for random number generation to ensure reproducibility.
        k: Range of random integers for generating features.
        a: Scaling factor for feature generation.
        c1, c2: Coefficients for potential feature interactions (not currently used).

    Returns:
        Tuple of numpy arrays (X, Y, EY) where:
        - X is the feature matrix of shape (N, 2 * dim) after extending with squares.
        - Y is the response variable vector of shape (N,).
        - EY is the expected value (without noise).
    """
    dim = len(beta)  # Get the number of features
    beta = np.array(beta, dtype=float)  # Convert beta to a numpy array
    X = np.ones((N, dim))  # Initialize feature matrix
    C = np.random.uniform(size=(dim, dim))  # Random transformation matrix

    # Generate independent features based on the specified range and scaling
    for i in range(dim):
        X[:, i] = np.random.randint(a * i, a * i + k, size=N) * 2

    X = X @ C  # Apply random transformation to the feature matrix

    # Extend feature matrix with squared terms
    X_ext = np.ones((N, 2 * dim))
    for i in range(dim):
        X_ext[:, i] = X[:, i]  # Original features
        X_ext[:, dim + i] = X[:, i] ** 2  # Squared features

    normal_noise = np.random.normal(0, sigma, size=N)  # Generate normal noise
    beta_ex = np.hstack((beta, beta / 2))  # Extend beta for squared features
    EY = np.matmul(X_ext, beta_ex) + b  # Calculate expected values with bias
    Y = EY + normal_noise  # Generate the response variable by adding noise

    return X, Y, EY  # Return the feature matrix, response variable, and expected values

def get_generate(N, beta):
    """Generate data and train a CatBoostRegressor model.

    Args:
        N: Number of samples to generate.
        beta: Coefficients for the linear model used in data generation.

    Returns:
        Trained CatBoostRegressor model.
    """
    # Generate simple data using the provided beta coefficients
    X, Y, EY = generate_simple_data(beta=beta, N=N, k=4, a=1, c1=2, c2=2, seed=1)

    # Initialize the CatBoostRegressor model with specified parameters
    model_genrate = CatBoostRegressor(
        iterations=50,
        depth=2,
        learning_rate=0.1,
        random_strength=1000,
        loss_function='RMSE',
        verbose=False,
        random_seed=1,
        feature_border_type='Median',
        score_function='L2',
    )

    # Create a copy of the feature matrix and drop the second feature by replacing it with its mean
    X_drop = X.copy()
    X_drop[:, 1] = np.mean(X_drop[:, 1])

    # Create a Pool object for CatBoost
    data_pool = Pool(X_drop, Y)

    # Fit the model to the data
    model_genrate.fit(data_pool)

    return model_genrate  # Return the trained model


def get_mse(model_genrate, N, sd, beta, nexp=20, plot=False):
    """Calculate mean squared error for a model over multiple experiments.

    Args:
        model_genrate: The pre-trained CatBoostRegressor model.
        N: Number of samples to generate for the initial data.
        sd: Standard deviation for noise in the response variable.
        beta: Coefficients for the linear model used in data generation.
        nexp: Number of experiments to run for error estimation.
        plot: Boolean indicating whether to plot the results.

    Returns:
        tN: Number of samples processed.
        res: List of results from each experiment.
        mean_res: Mean of the results from all experiments.
    """
    # Generate initial data using the provided beta coefficients
    X, Y, EY = generate_simple_data(beta=beta, N=N, k=4, a=1, c1=2, c2=2, seed=1)
    res = []

    # Generate test data
    X_t, Y_t, EY = generate_simple_data(beta=beta, N=2000, k=4, a=1, c1=2, c2=2, seed=1)
    X_drop = X_t.copy()
    X_drop[:, 1] = np.mean(X_drop[:, 1])
    
    # Create a Pool for the CatBoost model with dropped feature
    cal_pool = Pool(X_drop, Y_t)

    # Initialize and train a new CatBoostRegressor for calibration
    model_calk = CatBoostRegressor(
        iterations=2000,
        depth=2,
        learning_rate=0.01,
        random_strength=N**(5/4),
        loss_function='RMSE',
        verbose=False,
        random_seed=1,
        feature_border_type='Median',
        score_function='L2'
    )
    
    model_calk.fit(cal_pool)  # Train the calibration model

    # Predict with the generated model on initial data
    Y = model_genrate.predict(X)

    # Generate additional data for estimating noise and variance
    X_sigma, _, _ = generate_simple_data(beta=beta, N=10000, k=4, a=1, c1=2, c2=2)
    Y_sigma = model_genrate.predict(X_sigma) + beta[1] * X_sigma[:, 1]
    Y_sigma += np.random.normal(size=10000, scale=sd)  # Add noise

    # Prepare data for variance calculation
    X_sigma_drop = X_sigma.copy()
    X_sigma_drop[:, 1] = np.mean(X_sigma_drop[:, 1])
    yj = Y_sigma - model_genrate.predict(X_sigma_drop) - beta[1] * np.mean(X_sigma[:, 1])
    sigma = np.var(yj)  # Estimate variance

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    X_train_drop = X_train.copy()
    X_train_drop[:, 1] = np.mean(X_train_drop[:, 1])  # Drop feature by replacing it with mean
    k = cal_stationk(X_train_drop, model_calk)  # Calculate the kernel

    for i in range(nexp):
        tN, _, tmp = exp_wrapper(model_genrate, k, N, X, Y, sd=sd, beta=beta, sigma=sigma, plot=plot)
        print(tmp)
        res.append(tmp)  # Store the result of each experiment
        if _ == 0:
            print('error: T == 0')  # Check for error condition
        
    return tN, res, np.mean(res)  # Return total samples processed, results, and mean result










def exp_wrapper(model_genrate, k, N, X, Y, sd, beta, sigma, max_iter=4000, plot=False):
    """Run an experiment to evaluate a model's performance using a lazy learner.

    Args:
        model_genrate: The pre-trained CatBoostRegressor model.
        k: Kernel matrix used for lazy learning.
        N: Number of samples to process.
        X: Feature matrix.
        Y: Response variable.
        sd: Standard deviation for noise in the response variable.
        beta: Coefficients for the linear model used in data generation.
        sigma: Estimated variance from previous data.
        max_iter: Maximum number of iterations for optimization.
        plot: Boolean indicating whether to plot results during training.

    Returns:
        N: Number of samples processed.
        T: Optimal time index determined by the experiment.
        RMSE: Root Mean Squared Error from the validation set.
    """
    # Predict Y using the generated model
    Y = model_genrate.predict(X)

    # Add noise to the predictions
    Y += beta[1] * X[:, 1] + np.random.normal(size=N, scale=sd)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    train_pool = Pool(X_train, y_train)

    # Initialize and train a new CatBoostRegressor for full training
    model_full = CatBoostRegressor(
        iterations=max_iter,
        depth=2,
        learning_rate=0.1,
        random_strength=10000000,
        loss_function='RMSE',
        verbose=False,
        random_seed=1,
        feature_border_type='Median',
        score_function='L2'
    )

    model_full.fit(train_pool, eval_set=(X_test, y_test))  # Train the model with evaluation set

    N = X_train.shape[0]
    X_train_drop = X_train.copy()
    X_train_drop[:, 1] = np.mean(X_train_drop[:, 1])  # Drop feature by replacing it with mean
    X_test_drop = X_test.copy()
    X_test_drop[:, 1] = np.mean(X_test_drop[:, 1])

    # Prepare a training pool for lazy learning
    train_pool_drop_lazy = Pool(X_train_drop, y_train - model_full.predict(X_train_drop))
    lr = 1 / N  # Learning rate based on sample size

    # Initialize and train a lazy model
    model_lazy = CatBoostRegressor(
        iterations=4000,
        depth=2,
        learning_rate=lr,
        random_strength=N**(5/4),
        loss_function='RMSE',
        verbose=False,
        random_seed=1,
        feature_border_type='Median',
        score_function='L2'
    )

    # Calculate residuals for lazy model training
    yR = model_genrate.predict(X_train_drop) + beta[1] * X_train_drop[:, 1] - model_full.predict(X_train_drop)
    model_lazy.fit(train_pool_drop_lazy, eval_set=(X_train_drop, yR), use_best_model=False, plot=plot)

    # Prepare for optimization
    k = k * N
    yR = yR[:, None]  # Reshape yR for matrix operations
    kinv = np.linalg.pinv(k)  # Compute pseudoinverse of the kernel
    alpha = kinv @ yR  # Calculate weights
    R2 = yR.T @ kinv @ yR  # Compute R-squared value

    # Calculate constant for the optimization condition
    c = R2 * N / (4 * np.exp(1)**2 * sigma * lr**2)
    k = k / N

    while True:
        T = np.arange(1, max_iter)  # Time indices for optimization
        rhs = c / T**2  # Right-hand side of the inequality

        # Compute eigenvalues and ranks for the kernel
        e, s = np.linalg.eig(k)
        r = np.linalg.matrix_rank(k)
        er = np.real(e[:r])  # Real part of eigenvalues

        # Prepare left-hand side for comparison
        lhs = np.zeros((r, 2, len(T)))
        lhs[:, 0, :] = er[:, None]
        lhs[:, 1, :] = 1 / lr / T
        lhs = np.sum(np.min(lhs, axis=1), axis=0)  # Min across eigenvalues and lr

        # Find the optimal time index
        Tind = np.argmax(lhs > rhs)

        if Tind == 0:
            print('T find is 0')
            max_iter += 3000  # Increase max iterations if no suitable T found
            continue
        else:
            T = T[Tind] - 1  # Optimal time found
            break

    return N, T, model_lazy.get_evals_result()['validation']['RMSE'][Tind - 1]**2  # Return results


def tree_vi_exp_wrapper(X, Y, drop_i, sd, lr=0.1, es_lr=0.3, max_iter=2000, depth=3, esr=10, plot=False):
    """Evaluate the variance in predictions when dropping a feature using different training strategies.

    Args:
        X: Feature matrix.
        Y: Response variable.
        drop_i: Index of the feature to drop.
        sd: Random seed for reproducibility.
        lr: Learning rate for the full model.
        es_lr: Learning rate for the elastic model.
        max_iter: Maximum number of iterations for training.
        depth: Depth of the decision trees.
        esr: Early stopping rounds for training.
        plot: Boolean indicating whether to plot results during training.

    Returns:
        vi_est: Variance estimate from the elastic model.
        vi_retrain: Variance estimate from the retrained model.
        vi_drop: Variance estimate from dropping the feature.
        tes: Time taken to train the elastic model.
        tred: Time taken to train the retrained model.
    """
    # Split the data into training, validation, and test sets
    X_fit, X_test, y_fit, y_test = train_test_split(X, Y, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit, random_state=1)

    # Drop the specified feature by replacing it with its mean
    X_train_drop, X_val_drop, X_test_drop = dropdata(X_train, X_val, X_test, drop_i)

    # Prepare a training pool for the full model
    train_pool = Pool(X_train, y_train)
    model_full = CatBoostRegressor(
        iterations=max_iter,
        depth=depth,
        learning_rate=lr,
        random_strength=10000,
        loss_function='RMSE',
        verbose=False,
        random_seed=sd,
        feature_border_type='Median',
        score_function='L2',
    )

    # Fit the full model with early stopping
    model_full.fit(train_pool, eval_set=(X_val, y_val), early_stopping_rounds=esr, plot=plot, use_best_model=False)

    print(model_full.tree_count_)  # Print the number of trees in the full model

    # Calculate the mean squared error on the test set
    pre_full = np.mean((model_full.predict(X_test) - y_test.T[0, :]) ** 2)
    vi_drop = np.mean((model_full.predict(X_test_drop) - y_test.T[0, :]) ** 2) - pre_full

    N = X_train.shape[0]
    train_pool_red = Pool(X_train_drop, y_train)

    # Train the elastic model
    model_es = CatBoostRegressor(
        iterations=max_iter,
        depth=depth,
        learning_rate=es_lr,
        loss_function='RMSE',
        random_strength=10000,
        verbose=False,
        random_seed=sd,
        feature_border_type='Median',
        score_function='L2',
    )

    tmp = time.time()
    model_es.fit(train_pool_red, eval_set=(X_val_drop, y_val), init_model=model_full, early_stopping_rounds=esr, plot=plot, use_best_model=False)

    tes = time.time() - tmp  # Measure training time for the elastic model
    print(model_es.tree_count_ - model_full.tree_count_)  # Print the number of new trees added by the elastic model

    # Train the retrained model
    model_red = CatBoostRegressor(
        iterations=max_iter,
        depth=depth,
        learning_rate=lr,
        loss_function='RMSE',
        random_strength=10000,
        verbose=False,
        random_seed=sd,
        feature_border_type='Median',
        score_function='L2',
    )

    tmp = time.time()
    model_red.fit(train_pool_red, eval_set=(X_val_drop, y_val), early_stopping_rounds=esr, plot=plot, use_best_model=False)

    tred = time.time() - tmp  # Measure training time for the retrained model
    print(model_red.tree_count_)  # Print the number of trees in the retrained model

    # Calculate variance estimates for the models
    vi_est = np.mean((model_es.predict(X_test_drop) - y_test.T[0, :]) ** 2) - pre_full
    vi_retrain = np.mean((model_red.predict(X_test_drop) - y_test.T[0, :]) ** 2) - pre_full

    print('es', vi_est)
    print('retrain', vi_retrain)
    print('drop', vi_drop)

    return vi_est, vi_retrain, vi_drop, tes, tred  # Return variance estimates and training times
