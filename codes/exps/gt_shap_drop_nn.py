import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nn_utils import *

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

X = torch.tensor(X,dtype=torch.float32)
Y = torch.tensor(Y,dtype=torch.float32)
X = X - torch.mean(X,axis =0)
X = X / torch.std(X,axis = 0)
n_features = X.shape[1]



m = 2048

s = 50

p = n_features

widths = [p,m,m,1]
S = np.arange(p)
nexp = 10

via = np.zeros((p,nexp))



for i in range(p):
    drop_i = i
    tmpS = np.delete(S,drop_i)

    for j in range(nexp):
        #X,Y = generate_logistic_data(beta,N = 5000, corr = 0.5,seed = j)

        X_fit, X_test, y_fit, y_test = train_test_split(X, Y,random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit,random_state=1)
        X_fit_drop = X_fit.clone()
        X_fit_drop[:,drop_i] = torch.mean(X_fit_drop[:,drop_i])

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        dm = FlexDataModule( X_train, y_train, X_val, y_val)
        #train full model
        lr = 0.1

        full_nn = LazyNet(widths,lr = lr)
        full_nn.reset_parameters()


        # train full network
        early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
        cb = MetricTracker()
        trainer = pl.Trainer(callbacks=[cb,early_stopping], max_epochs=800,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(full_nn, dm)
        print(trainer.current_epoch)

        allm = list(powerset(tmpS))

        w = [  1/p /  scipy.special.binom(p-1, p-1-len(v)) for v in allm]

        vis = np.zeros(s)
        for t in  range(s):
            tmp_s = np.random.choice(len(allm), p = w)

            # if tmp_s:
            tmp_s = allm[tmp_s]
            dropS = np.delete(S,[drop_i] + list(tmp_s))


            X_train_drop, X_val_drop, X_test_drop = dropdata(X_train,X_val, X_test,dropS)
         
            X_train_drop2 = X_train_drop.clone()
            X_val_drop2 = X_val_drop.clone()
            X_test_drop2 = X_test_drop.clone()
            X_train_drop2[:,drop_i] = torch.mean(X_train_drop2[:,drop_i])
            X_val_drop2[:,drop_i] = torch.mean(X_val_drop2[:,drop_i])
            X_test_drop2[:,drop_i] = torch.mean(X_test_drop2[:,drop_i])

            vi_est = torch.mean((full_nn(X_test_drop2) - y_test)**2)  -  torch.mean((full_nn(X_test_drop) - y_test)**2) 



                #vi_est = torch.mean((lazy_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 
            vis[t] = vi_est
            # else:
            #     vis[i,t] = 
    
        via[i,j] = np.mean(vis)


np.savetxt('via_drop_nn_gt.txt',via)



