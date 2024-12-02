import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils_gdbt import *

import time
from sklearn.model_selection import KFold


beta = [5,4,3,2,1]  +  list(np.zeros(95))





p = 100

widths = [p,m,1]

drop_i = 0


drop_i = 0
t1 = np.zeros(10)
t2 =  np.zeros(10)

res = np.zeros(10)
res2 = np.zeros(10)
res3 = np.zeros(10)
depth = 3



for t in range(10):

    X,Y=generate_linear_data(beta , N= 5000,corr = 0.5)
 
    
    Y = Y.reshape(-1,1)
    vi_est, vi_retrain, vi_drop, tes, tred = tree_vi_exp_wrapper(X,Y, drop_i,sd = t,lr = 0.1, es_lr=0.1, esr= 25,depth = depth,max_iter=1000, plot =False)
  
    t1[t] = tes
    t2[t] = tred
    res[t] = vi_est
    res2[t] = vi_retrain
    res3[t] = vi_drop
    print( np.abs(vi_est-vi_retrain) < np.abs(vi_drop-vi_retrain))


np.savetxt('../results/hdp_tree_es.txt',res)
np.savetxt('../results/hdp_tree_retrain.txt',res2)
np.savetxt('../results/hdp_tree_drop.txt',res3)
np.savetxt('../results/hdp_tree_es_time.txt',t1)
np.savetxt('../results/hdp_tree_retrain_time.txt',t2)

    