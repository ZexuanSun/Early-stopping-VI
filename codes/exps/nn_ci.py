import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nn_utils import *

import time
from sklearn.model_selection import KFold


from itertools import chain, combinations
import scipy.special




ho = 0.5
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



        p = beta.shape[0]

        widths = [p,m,m, 1]

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
        early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
        cb = MetricTracker()
        trainer = pl.Trainer(callbacks=[cb,early_stopping], max_epochs=800,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(full_nn, dm)


        dropS = [0]
        X_train_drop, X_val_drop, X_test_drop = dropdata(X_train,X_val, X_test,dropS)
        lazy_nn = LazyNet(widths,lr = 0.1)
        lazy_nn.init_parameters(full_nn)


        dm_lazy = FlexDataModule( X_train_drop, y_train,  X_val_drop,  y_val)
        early_stopping = EarlyStopping('val_loss', min_delta=1e-3)
        cb = MetricTracker()

        trainer = pl.Trainer(callbacks=[cb,early_stopping], max_epochs=100,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(lazy_nn, dm_lazy)
        print(trainer.current_epoch)


        vi_est = torch.mean((lazy_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 
        vi_drop =  torch.mean((full_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 

        vi_se = torch.sqrt(torch.var((lazy_nn(X_test_drop) - y_test)**2 - (full_nn(X_test) - y_test)**2)/y_test.shape[0])
        tmp_ci_l = (vi_est - 1.96*vi_se).item()
        tmp_ci_r = (vi_est + 1.96*vi_se).item()
        print( tmp_ci_l,tmp_ci_r)
        ci_l[i,j] = tmp_ci_l 
        ci_r[i,j] = tmp_ci_r

np.savetxt('nn_ci_L.txt',ci_l)
np.savetxt('nn_ci_r.txt',ci_r)
    
    
    