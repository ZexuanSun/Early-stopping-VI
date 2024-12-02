import numpy as np
from catboost import Pool, CatBoostRegressor
import random
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import Pool, cv
from utils_gdbt import *

beta = np.array([2, 3, 4])
sd = 5
model_gerate = get_generate(N=100, beta=beta)
nexp = 10
expN = np.arange(200, 2300, 300)
res = np.zeros((nexp, len(expN)))

print('test')

loss = []
Nl = []

for j in range(len(expN)):
    print(j)
    tN, tloss, tmp = get_mse(model_genrate=model_gerate, N=expN[j], sd=sd, beta=beta, nexp=nexp, plot=False)
    res[:, j] = tloss
    Nl.append(tN)

np.savetxt('tree_sim_loss_32.txt', res)
np.savetxt('tree_sim_N_32.txt', Nl)
