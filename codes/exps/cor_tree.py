import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *

import time
from sklearn.model_selection import KFold


from itertools import chain, combinations
import scipy.special





nexp = 10

beta = np.array([1.5, 1.2, 1, 0, 0, 0])
rhos = np.linspace(0,1,300)
res  = np.zeros((len(rhos),nexp))
resd = np.zeros((len(rhos),nexp))




for i in range(len(rhos)):
    for j in range(nexp):

        X, Y = generate_linear_data(beta=beta, sigma= 0.1, corr=rhos[i])

        drop_i = 0 

        X_fit, X_test, y_fit, y_test = train_test_split(X, Y,random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit,random_state=1)
        X_fit_drop = np.copy(X_fit)
        X_fit_drop[:,drop_i] = np.mean(X_fit_drop[:,drop_i])

        max_iter = 1000

        depth = 3
        lr = 0.1
        es_lr = 0.1
        plot = False



        train_pool = Pool(X_train,y_train)
        model_full= CatBoostRegressor(iterations=max_iter,
                                    depth= depth,
                                    learning_rate= lr,
                                    random_strength= 10000,
                                    loss_function='RMSE',
                                    verbose=False,
                                    feature_border_type='Median',
                                    score_function='L2',            
                                    )
        model_full.fit(train_pool,eval_set=(X_val, y_val), early_stopping_rounds = 10,plot = plot)
        pre_full = np.mean((model_full.predict(X_test) - y_test)**2)


        dropS = [0]

        X_train_drop, X_val_drop, X_test_drop = dropdata(X_train,X_val, X_test,dropS)

        vi_drop = np.mean((model_full.predict(X_test_drop) - y_test)**2) - pre_full
        resd[i,j] = vi_drop

     
        train_pool_red = Pool(X_train_drop,y_train)

        model_es= CatBoostRegressor(iterations=max_iter,
                                                    depth= depth,
                                                    learning_rate= es_lr,
                                                    loss_function='RMSE',
                                                    random_strength= 10000,
                                                    verbose=False,
                                                    feature_border_type='Median',
                                                    score_function='L2'
                                                    )
                    
        model_es.fit(train_pool_red,eval_set=(X_val_drop, y_val), init_model= model_full,   early_stopping_rounds = 10,plot = plot)

            
        vi_est =  np.mean((model_es.predict(X_test_drop) - y_test)**2) - pre_full
        res[i,j] = vi_est


np.savetxt('cor_nn_exp_es_tree.txt',res)

np.savetxt('cor_nn_exp_drop_tree.txt',resd)






