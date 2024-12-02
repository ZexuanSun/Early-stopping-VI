import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *
import time
from sklearn.model_selection import train_test_split
from itertools import chain, combinations
import scipy.special

beta = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
p = 10
S = np.arange(10)
nexp = 10
via_es = np.zeros((p, nexp))
via_drop = np.zeros((p, nexp))
via_retrain = np.zeros((p, nexp))
s = 50
max_iter = 1000

depth = 3
lr = 0.1
es_lr = 0.1
tes = np.zeros((p, nexp))
tre = np.zeros((p, nexp))
plot = False

# Loop over each feature to be dropped
for i in range(p):
    drop_i = i  # Feature index to drop
    tmpS = np.delete(S, drop_i)  # Remaining features after dropping

    for j in range(nexp):  # Loop through each experiment
        t1 = time.time()  # Start time for the full model training
        while True:
            # Generate logistic data
            X, Y = generate_logistic_data(beta, N=5000, corr=0.5)

            # Split the data into training, validation, and test sets
            X_fit, X_test, y_fit, y_test = train_test_split(X, Y)
            X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit)

            # Prepare data for dropping the feature
            X_fit_drop = np.copy(X_fit)
            X_fit_drop[:, drop_i] = np.mean(X_fit_drop[:, drop_i])  # Replace dropped feature with mean

            lr = 0.1  # Reset learning rate for each experiment

            # Train the full model
            train_pool = Pool(X_train, y_train)
            model_full = CatBoostRegressor(iterations=max_iter,
                                            depth=depth,
                                            learning_rate=lr,
                                            random_strength=10000,
                                            loss_function='RMSE',
                                            verbose=False,
                                            random_seed=1,
                                            feature_border_type='Median',
                                            score_function='L2',)
            try:
                model_full.fit(train_pool, eval_set=(X_val, y_val), early_stopping_rounds=10, plot=plot)
                break  # Exit loop if training is successful
            except:
                continue  # Retry if an error occurs

        tfull = time.time() - t1  # Total time for full model training

        # Calculate performance metric (MSE)
        pre_full = np.mean((model_full.predict(X_test) - y_test) ** 2)
        print(model_full.tree_count_)  # Output the number of trees in the full model

        # Generate all combinations of the remaining features
        allm = list(powerset(tmpS))

        # Calculate weights for sampling combinations
        w = [1 / p / scipy.special.binom(p - 1, p - 1 - len(v)) for v in allm]

        # Arrays to store variance estimates
        vis_es = np.zeros(s)
        vis_drop = np.zeros(s)
        vis_retrain = np.zeros(s)
        t_es = 0  # Total time for elastic model training
        t_re = 0  # Total time for retrained model training

        for t in range(s):  # Loop over sample size
            while True:
                tmp_s = np.random.choice(len(allm), p=w)  # Sample a combination based on weights
                dropS = np.delete(S, [drop_i] + list(tmp_s))  # Create a list of features to drop
                if len(dropS) < 9:  # Ensure at least one feature is retained
                    break
            print(dropS)

            # Prepare the data by dropping selected features
            X_train_drop, X_val_drop, X_test_drop = dropdata(X_train, X_val, X_test, dropS)

            # Train the elastic model
            train_pool_red = Pool(X_train_drop, y_train)

            print('es model train')
            model_es = CatBoostRegressor(iterations=max_iter,
                                          depth=depth,
                                          random_strength=10000,
                                          learning_rate=es_lr,
                                          loss_function='RMSE',
                                          verbose=False,
                                          random_seed=1,
                                          feature_border_type='Median',
                                          score_function='L2')
            f1 = time.time()
            model_es.fit(train_pool_red, eval_set=(X_val_drop, y_val), init_model=model_full,
                          early_stopping_rounds=10, plot=plot)
            t_es += time.time() - f1  # Total time for elastic model training

            print(model_es.tree_count_ - model_full.tree_count_)  # Output additional trees added by the elastic model

            # Prepare dropped data for second elastic model
            X_train_drop2 = np.copy(X_train_drop)
            X_val_drop2 = np.copy(X_val_drop)
            X_test_drop2 = np.copy(X_test_drop)
            X_train_drop2[:, drop_i] = np.mean(X_train_drop2[:, drop_i])
            X_val_drop2[:, drop_i] = np.mean(X_val_drop2[:, drop_i])
            X_test_drop2[:, drop_i] = np.mean(X_test_drop2[:, drop_i])

            train_pool_red2 = Pool(X_train_drop2, y_train)

            # Train the second elastic model
            model_es2 = CatBoostRegressor(iterations=max_iter,
                                           depth=depth,
                                           random_strength=10000,
                                           learning_rate=es_lr,
                                           loss_function='RMSE',
                                           verbose=False,
                                           random_seed=1,
                                           feature_border_type='Median',
                                           score_function='L2')
            f1 = time.time()
            model_es2.fit(train_pool_red2, eval_set=(X_val_drop2, y_val), init_model=model_full,
                           early_stopping_rounds=10, plot=plot)
            t_es += time.time() - f1  # Total time for second elastic model training

            print(model_es2.tree_count_ - model_full.tree_count_)  # Output additional trees added by the second elastic model

            # Calculate variance estimates
            vi_est = np.mean((model_es2.predict(X_test_drop2) - y_test) ** 2) - \
                      np.mean((model_es.predict(X_test_drop) - y_test) ** 2)

            vi_drop = np.mean((model_full.predict(X_test_drop2) - y_test) ** 2) - \
                       np.mean((model_full.predict(X_test_drop) - y_test) ** 2)

            print('retrain')

            # Train the retrained model
            model_red = CatBoostRegressor(iterations=max_iter,
                                           depth=depth,
                                           learning_rate=lr,
                                           loss_function='RMSE',
                                           verbose=False,
                                           random_strength=10000,
                                           random_seed=1,
                                           feature_border_type='Median',
                                           score_function='L2')

            f1 = time.time()
            model_red.fit(train_pool_red, eval_set=(X_val_drop, y_val), early_stopping_rounds=10, plot=plot)
            t_re += time.time() - f1  # Total time for retrained model training

            print(model_red.tree_count_)  # Output the number of trees in the retrained model

            # Prepare dropped data for second retrained model
            model_red2 = CatBoostRegressor(iterations=max_iter,
                                            depth=depth,
                                            learning_rate=lr,
                                            loss_function='RMSE',
                                            verbose=False,
                                            random_strength=10000,
                                            random_seed=1,
                                            feature_border_type='Median',
                                            score_function='L2')

            f1 = time.time()
            model_red2.fit(train_pool_red2, eval_set=(X_val_drop2, y_val), early_stopping_rounds=10, plot=plot)
            t_re += time.time() - f1  # Total time for second retrained model training

            print(model_red2.tree_count_)  # Output the number of trees in the second retrained model

            # Calculate variance estimates for retrained model
            vi_retrain = np.mean((model_red2.predict(X_test_drop2) - y_test) ** 2) - \
                          np.mean((model_red.predict(X_test_drop) - y_test) ** 2)

            # Store variance estimates
            vis_es[t] = vi_est
            vis_drop[t] = vi_drop
            vis_retrain[t] = vi_retrain

            print(np.abs(vi_est - vi_retrain) < np.abs(vi_drop - vi_retrain))  # Comparison of variance estimates

            print('drop', vi_drop)
            print('es', vi_est)
            print('retrain', vi_retrain)

        # Store results for the current experiment
        tes[i, j] = tfull + t_es
        tre[i, j] = t_re
        via_es[i, j] = np.mean(vis_es)
        via_drop[i, j] = np.mean(vis_drop)
        via_retrain[i, j] = np.mean(vis_retrain)


np.savetxt('via_tree_es_time.txt', tes)
np.savetxt('via_tree_retrain_time.txt', tre)
np.savetxt('via_es_tree_new.txt', via_es)
np.savetxt('via_drop_tree_new.txt', via_drop)
np.savetxt('via_retrain_tree_new.txt', via_retrain)
