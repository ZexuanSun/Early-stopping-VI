import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *

import time
from sklearn.model_selection import KFold


from itertools import chain, combinations
import scipy.special




beta = np.array([1.5, 1.2, 1, 0, 0, 0])
rhos = [0.0,0.2,0.5,0.8,1.0]
nexp = 100
ci_l = np.zeros((len(rhos),nexp))
ci_r = np.zeros((len(rhos),nexp))

drop_i = 0

m = 2048

for i in range(len(rhos)):
    rho = rhos[i]
    for j in range(nexp):
        X, Y = generate_linear_data(beta=beta, sigma= 0.1, corr=rho)

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


        train_pool_red = Pool(X_train_drop,y_train)

        model_es= CatBoostRegressor(iterations=max_iter,
                                                    depth= depth,
                                                    learning_rate= es_lr,
                                                    random_strength= 10000,
                                                    loss_function='RMSE',
                                                    verbose=False,
                                                    feature_border_type='Median',
                                                    score_function='L2'
                                                    )
                    
        model_es.fit(train_pool_red,eval_set=(X_val_drop, y_val), init_model= model_full,   early_stopping_rounds = 10,plot = plot)
        vi_est =  np.mean((model_es.predict(X_test_drop) - y_test)**2) - pre_full
        vi_sd =   np.sqrt(np.var((model_es.predict(X_test_drop) - y_test)**2 - (model_full.predict(X_test) - y_test)**2)/y_test.shape[0])
        ci = (vi_est - 1.96*vi_sd, vi_est+ 1.96*vi_sd)

        tmp_ci_l = ci[0]
        tmp_ci_r = ci[1]
        print( tmp_ci_l,tmp_ci_r)
        ci_l[i,j] = tmp_ci_l 
        ci_r[i,j] = tmp_ci_r

np.savetxt('tree_ci_L.txt',ci_l)
np.savetxt('tree_ci_r.txt',ci_r)
    
    
    