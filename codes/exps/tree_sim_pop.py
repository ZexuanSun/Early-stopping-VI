

import numpy as np
from catboost import Pool, CatBoostRegressor


import random
import numpy as np
import random
from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import Pool, cv
from utils_gdbt import *


print('test test test')


beta = np.array([2,3,4]) 
sd = 5
model_gerate = get_generate(N=100,beta = beta)
nexp = 10
expN =  np.arange(100,3000,300)
#expN = np.arange(200,3000,200)
# expN = np.arange(200,2300,300)
#expN = np.array([5000,6000,7000])
#expN = [200]
res =  np.zeros((nexp,len(expN)))

print('test')

loss = []

Nl = []


for j in range(9):
    print(j)
    
    #beta = np.array([2,3]) 
    tN,tloss,tmp = get_mse( model_genrate = model_gerate, N = expN[j],sd = sd,beta = beta,nexp = nexp,seed =j, plot = False)
    res[:,j] = tloss
    Nl.append(tN)

np.savetxt('tree_sim_loss_pop.txt',res)
np.savetxt('tree_sim_N_pop.txt',Nl)
    