import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *

import time
from sklearn.model_selection import KFold


from itertools import chain, combinations
import scipy.special







bsd = pd.read_csv('gt_2015.csv',sep=',')


features = np.arange(0,9)

X = bsd.iloc[:,features].values
Y = bsd.iloc[:,-2].values
Y = Y.reshape(-1,1)
n_features = X.shape[1]

n_features = X.shape[1]
X_o = np.copy(X)
Y_o = np.copy(Y)


p = n_features
S = np.arange(p)
nexp = 10
via_es = np.zeros((p,nexp))
via_drop = np.zeros((p,nexp))
via_retrain = np.zeros((p,nexp))


s = 50
max_iter = 1000


depth = 8
lr = 0.1
es_lr = 0.1

plot = False
#for i in range(p):

for i in range(9):
    drop_i = i
    tmpS = np.delete(S,drop_i)

    for j in range(nexp):
        X = np.copy(X_o)
        Y = np.copy(Y_o)
        #X,Y = generate_logistic_data(beta,N =5000, corr = 0.5)

        X_fit, X_test, y_fit, y_test = train_test_split(X, Y,random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit,random_state=1)
        X_fit_drop = np.copy(X_fit)
        X_fit_drop[:,drop_i] = np.mean(X_fit_drop[:,drop_i])

     
        lr = 0.1

        train_pool = Pool(X_train,y_train)
        model_full= CatBoostRegressor(iterations=max_iter,
                                        depth= depth,
                                        learning_rate= lr,
                                        random_strength= 10000,
                                        loss_function='RMSE',
                                        verbose=False,
                                        random_seed=1,
                                        feature_border_type='Median',
                                        score_function='L2',            
                                        )
        model_full.fit(train_pool,eval_set=(X_val, y_val), early_stopping_rounds = 10,plot = plot,use_best_model=False)
        pre_full = np.mean((model_full.predict(X_test) - y_test)**2)
        

        print(model_full.tree_count_)



        allm = list(powerset(tmpS))

        w = [  1/p /  scipy.special.binom(p-1, p-1-len(v)) for v in allm]

        vis_es= np.zeros(s)
        vis_drop = np.zeros(s)
        vis_retrain = np.zeros(s)
        for t in  range(s):
            # tmp_s = np.random.choice(len(allm), p = w)

            # # if tmp_s:
            # tmp_s = allm[tmp_s]
            while True:
                tmp_s = np.random.choice(len(allm), p = w)

            # if tmp_s:
                tmp_s = allm[tmp_s]
                dropS = np.delete(S,[drop_i] + list(tmp_s))
                if len(dropS) < n_features - 1:
                    break
            print(dropS)


            X_train_drop, X_val_drop, X_test_drop = dropdata(X_train,X_val, X_test,dropS)

            X_train_drop2 = np.copy(X_train_drop)
            X_val_drop2 = np.copy(X_val_drop)
            X_test_drop2 = np.copy(X_test_drop)
            X_train_drop2[:,drop_i] = np.mean(X_train_drop2[:,drop_i])
            X_val_drop2[:,drop_i] = np.mean(X_val_drop2[:,drop_i])
            X_test_drop2[:,drop_i] = np.mean(X_test_drop2[:,drop_i])



            vi_drop = np.mean((model_full.predict(X_train_drop2) - y_train[:,0])**2) - np.mean((model_full.predict(X_train_drop) - y_train[:,0])**2)
            vi_drop = np.mean((model_full.predict(X_test_drop2) - y_test[:,0])**2) - np.mean((model_full.predict(X_test_drop) - y_test[:,0])**2)


            train_pool_red = Pool(X_train_drop,y_train)

            print('es model train')


            model_es= CatBoostRegressor(iterations=max_iter,
                                            depth= depth,
                                            learning_rate= es_lr,
                                            random_strength= 10000,
                                            loss_function='RMSE',
                                            verbose=False,
                                            random_seed=1,
                                            feature_border_type='Median',
                                            score_function='L2'
                                            )
            
            model_es.fit(train_pool_red,eval_set=(X_val_drop, y_val), init_model= model_full,   early_stopping_rounds = 10,plot = plot,use_best_model=False)

     
            print(model_es.tree_count_ - model_full.tree_count_ )





            

            train_pool_red2 = Pool(X_train_drop2,y_train)


            model_es2= CatBoostRegressor(iterations=max_iter,
                                            depth= depth,
                                            learning_rate= es_lr,
                                            random_strength= 10000,
                                            loss_function='RMSE',
                                            verbose=False,
                                            random_seed=1,
                                            feature_border_type='Median',
                                            score_function='L2'
                                            )
            
            model_es2.fit(train_pool_red2,eval_set=(X_val_drop2, y_val), init_model= model_full,   early_stopping_rounds = 10,plot = plot,use_best_model=False)

     
            print(model_es2.tree_count_ - model_full.tree_count_ )


            #vi_est =   np.mean((model_es2.predict(X_train_drop2) - y_train[:,0])**2) - np.mean((model_es.predict(X_train_drop) - y_train[:,0])**2)
            vi_est =   np.mean((model_es2.predict(X_test_drop2) - y_test[:,0])**2) - np.mean((model_es.predict(X_test_drop) - y_test[:,0])**2)


            







            print('retrain')



            model_red = CatBoostRegressor(iterations=max_iter,
                                    depth= depth,
                                    learning_rate= lr,
                                    loss_function='RMSE',
                                    verbose=False,
                                    random_strength= 10000,
                                    random_seed=1,
                                    feature_border_type='Median',
                                    score_function='L2'
                                    )

            model_red.fit(train_pool_red,eval_set=(X_val_drop, y_val),early_stopping_rounds = 10,plot = plot,use_best_model=False)


            print(model_red.tree_count_)


            model_red2 = CatBoostRegressor(iterations=max_iter,
                                    depth= depth,
                                    learning_rate= lr,
                                    loss_function='RMSE',
                                    random_strength= 10000,
                                    verbose=False,
                                    random_seed=1,
                                    feature_border_type='Median',
                                    score_function='L2'
                                    )

            model_red2.fit(train_pool_red2,eval_set=(X_val_drop2, y_val),early_stopping_rounds = 10,plot = plot,use_best_model=False)


            print(model_red2.tree_count_)

            #vi_retrain = np.mean((model_red2.predict(X_train_drop2) - y_train[:,0])**2) - np.mean((model_red.predict(X_train_drop) - y_train[:,0])**2)
            vi_retrain = np.mean((model_red2.predict(X_test_drop2) - y_test[:,0])**2) - np.mean((model_red.predict(X_test_drop) - y_test[:,0])**2)





            vis_es[t] = vi_est
            
            vis_drop[t] = vi_drop
            vis_retrain[t] = vi_retrain
    

            print( np.abs(vi_est-vi_retrain)   < np.abs(vi_drop-vi_retrain))

            print('drop',vi_drop)
            print('es',vi_est)

   
            print('retrain',vi_retrain)

        via_es[i,j] = np.mean(vis_es)
        via_drop[i,j] = np.mean(vis_drop)
        via_retrain[i,j] = np.mean(vis_retrain)



np.savetxt('via_es_tree_gt.txt',via_es)
np.savetxt('via_drop_tree_gt.txt',via_drop)
np.savetxt('via_retrain_tree_gt.txt',via_retrain)
