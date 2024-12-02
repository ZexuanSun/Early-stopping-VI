import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *

import time
from sklearn.model_selection import KFold
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

bsd = pd.read_csv('gt_2015.csv',sep=',')


features = np.arange(0,9)

X = bsd.iloc[:,features].values
Y = bsd.iloc[:,-2].values
X = X - np.mean(X,axis =0)
X = X / np.std(X,axis = 0)
Y = Y.reshape(-1,1)
n_features = X.shape[1]


depth = 8
nexp = 10
n_features = X.shape[1]

drop_i = 0
nexp  = 10
es = np.zeros((n_features,nexp))
retrain = np.zeros((n_features,nexp))
drop = np.zeros((n_features,nexp))


for j in range(n_features):
    print(j)
    for i in range(nexp):

        #X,Y=generate_2lnn_data(W,V,n=1000,corr=0.05)
        #X,Y=generate_logistic_data(beta , N= 1000,corr = 0.05)
        vi_est, vi_retrain, vi_drop, tes, tred = tree_vi_exp_wrapper(X,Y, drop_i = j, sd = i , lr = 0.1, es_lr=0.1, esr= 10,depth = depth,max_iter=3000, plot =False)
        es_error = np.abs(vi_est - vi_retrain) / np.abs(vi_retrain)
        drop_error = np.abs(vi_drop - vi_retrain) / np.abs(vi_retrain)
        print(es_error < drop_error)
        es[j,i] = vi_est
        retrain[j,i] = vi_retrain
        drop[j,i] = vi_drop


np.savetxt('../results/gt_es_tree.txt',es)
np.savetxt('../results/gt_retrain_tree.txt',retrain)
np.savetxt('../results/gt_drop_tree.txt',drop)