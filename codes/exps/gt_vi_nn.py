import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nn_utils import *

import time
from sklearn.model_selection import KFold
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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




n_features = X.shape[1]
widths = [n_features,2048,2048,1]

nexp  = 10
es = np.zeros((n_features,nexp))
retrain = np.zeros((n_features,nexp))
drop = np.zeros((n_features,nexp))
for j in range(n_features):
    drop_i = j
    for i in range(nexp):
        es_vi, retrain_vi, drop_vi = vi_exp_wrapper(X,Y,drop_i,widths)
        es[j,i] = es_vi
        retrain[j,i] = retrain_vi
        drop[j,i] = drop_vi


np.savetxt('gt_red_es.txt',es)
np.savetxt('gt_red_retrain.txt',retrain)
np.savetxt('gt_red_drop.txt',drop)