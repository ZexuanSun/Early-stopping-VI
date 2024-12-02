import warnings
warnings.filterwarnings("ignore")
import numpy as np
from lazynetwork import *
import matplotlib.pyplot as plt
from nn_utils import *
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

beta = np.array([3,4,5])
Ns = np.arange(200,2300,300)
p = beta.shape[0]
nexp = 10
widths = [p,2048,1]
nl = []
res = np.zeros((nexp,len(Ns)))

for i in range(len(Ns)):
    _,gen_nn = create_gen(beta,widths)
    X, Y = generate_linear_data(beta=beta,N = Ns[i],corr=0.0)
    for j in range(nexp):
        widths = [p,2048,1]
        nt, T , loss= nn_mse_exp(X,sd = 5,widths=widths,beta = beta, gen_nn=gen_nn)
        res[j,i] = loss
    nl.append(nt)

np.savetxt('nn_sim_res_2048_32.txt',res)
np.savetxt('nn_sim_N_2048_32.txt',nl)
