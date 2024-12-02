





import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nn_utils import *

import time
from sklearn.model_selection import KFold

from data_generating_funcs import FlexDataModule2

from networks import NN4vi
from lazyvi import LazyVI

from utils import dropout



beta = [5,4,3,2,1] + list(np.zeros(95))


W,V=generate_WV(beta, 12, V='random', sigma=0.1)


X,Y=generate_2lnn_data(W,V,n=1000,corr=0.5)


m = 2048

p = 100

widths = [p,m,m,1]

drop_i = 0
n_k = 3

drop_i = 0
nexp = 10
t1 = np.zeros(nexp)
t2 =  np.zeros(nexp)
t3 = np.zeros(nexp)

res = np.zeros(nexp)
res2 = np.zeros(nexp)
res3 = np.zeros(nexp)
res4 = np.zeros(nexp)
for t in range(nexp):
    X,Y=generate_2lnn_data(W,V,n=5000,corr=0.5)

    dm2 = FlexDataModule2(X,Y)


    full_nn2 = NN4vi(p, [m,m], 1)

    # train full network
    early_stopping = EarlyStopping('val_loss', min_delta=1e-5)
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=1000)
    trainer.fit(full_nn2, dm2)

    # extract train/test response for measuring performance
    y_train = dm2.train.dataset.tensors[1]
    y_test = dm2.test.tensors[1]
    X_test = dm2.test.tensors[0]

    # create a modified dataset with the first variable dropped out
    j = 0
    Xj = dropout(X, j)
    dmj = FlexDataModule2(Xj, Y)
    dmj.setup()
    Xj_train = dmj.train.dataset.tensors[0]
    Xj_test = dmj.test.tensors[0]


   
    lv = LazyVI(full_nn2, lambda_path=np.logspace(1,3,20)) 
    t0 = time.time()# initialize module with fully trained network, define regularization path
    lv.fit(Xj_train, y_train) # fit LazyVI
    tlazy = time.time() - t0
    t3[t] = tlazy

    lazy_vi, lazy_se = lv.calculate_vi(X_test, Xj_test, y_test, se=True)
    res4[t] = lazy_vi



    X_fit, X_test, y_fit, y_test = train_test_split(X, Y,random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit,random_state=1)
    X_fit_drop = X_fit.clone()
    X_fit_drop[:,drop_i] = torch.mean(X_fit_drop[:,drop_i])



    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)



    X_train_drop = X_train.clone()
    X_train_drop[:,drop_i] =  torch.mean(X_train_drop[:,drop_i])

    X_val_drop = X_val.clone()
    X_val_drop[:,drop_i] = torch.mean(X_val_drop[:,drop_i])


    X_train_drop = X_train.clone()
    X_train_drop[:,drop_i] =  torch.mean(X_train_drop[:,drop_i])

    X_val_drop = X_val.clone()
    X_val_drop[:,drop_i] = torch.mean(X_val_drop[:,drop_i])

    X_test_drop = X_test.clone()
    X_test_drop[:,drop_i] = torch.mean(X_test_drop[:,drop_i])




    #print(torch.var(y_train - X_train_drop @ beta[:,None]))


    dm = FlexDataModule( X_train, y_train, X_val, y_val)
    lr = 0.1

    full_nn = LazyNet(widths,lr = lr)
    full_nn.reset_parameters()


    # train full network
    early_stopping = EarlyStopping('val_loss', min_delta=1e-5)
    cb = MetricTracker()
    trainer = pl.Trainer(callbacks=[cb,early_stopping], max_epochs=800,enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(full_nn, dm)
    print(trainer.current_epoch)


   






    lr = 0.1

    max_iter = 10

   

    res3[t] =  torch.mean((full_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 


    lazy_nn = LazyNet(widths,lr = lr)
    lazy_nn.init_parameters(full_nn)

    #dm = FlexDataModule( X_train_drop, y_train, X_val_drop, y_val)
    #dm = FlexDataModule( X_train_drop, y_train, X_train_drop, gen_nn(X_train_drop) - init_nn(X_train_drop))
    dm_lazy = FlexDataModule( X_train_drop, y_train,   X_val_drop,  y_val)
    #cb = MetricTracker()
    early_stopping = EarlyStopping('val_loss', min_delta=1e-3)

    trainer = pl.Trainer(callbacks=[early_stopping],enable_progress_bar=False,enable_model_summary=False)
    tmp = time.time()
    trainer.fit(lazy_nn, dm_lazy)
    t1[t] = time.time() - tmp

    vi_est = torch.mean((lazy_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 

    res[t] = vi_est


   


    red_nn = LazyNet(widths,lr = lr)
    red_nn.reset_parameters()

    dm_red = FlexDataModule( X_train_drop, y_train, X_val_drop, y_val)
    # train full network
    early_stopping = EarlyStopping('val_loss', min_delta=1e-5)
    cb = MetricTracker()
    trainer = pl.Trainer(callbacks=[cb,early_stopping], max_epochs=100,enable_progress_bar=False,enable_model_summary=False)
    tmp = time.time()
    trainer.fit(red_nn, dm_red)
    t2[t] = time.time() - tmp

    print(trainer.current_epoch)
    vi_retrain = torch.mean((red_nn(X_test_drop) - y_test)**2)  -  torch.mean((full_nn(X_test) - y_test)**2) 
    res2[t] = vi_retrain





    print(vi_est)
    print(vi_retrain)


np.savetxt('hdlazy_new.txt',res)

np.savetxt('hdretrain_new.txt',res2)
np.savetxt('hdlazyvi_new.txt',res4)



np.savetxt('hdlazyvi_time_new.txt',t3)


np.savetxt('hdlazy_time_new.txt',t1)

np.savetxt('hdretrain_time_new.txt',t2)


np.savetxt('hddrop_new.txt',res3)


