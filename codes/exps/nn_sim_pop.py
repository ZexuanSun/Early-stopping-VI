import warnings

warnings.filterwarnings("ignore")
import numpy as np
from lazynetwork import *
import matplotlib.pyplot as plt
from nn_utils import *
import torch.nn as nn
#from utils_gdbt import *
from sklearn.model_selection import train_test_split
# generate data
from sklearn.model_selection import KFold



beta = np.array([3,4,5])

# Ns = np.array([5000,6000,7000])
# Ns = np.arange(100,3000,300)
Ns = np.arange(200,2300,300)
p = beta.shape[0]
nexp = 10
#widths = [p,1024,1]
widths = [p,2048,1]
nl = []

for q in range(10):
    res = np.zeros((nexp,len(Ns)))
    for i in range(len(Ns)):
        _,gen_nn = create_gen(beta,widths)
        #X, Y = generate_linear_data(beta=beta,N = Ns[i],corr=0.0)
        for j in range(nexp):

            #widths = [p,1024,1]
            widths = [p,2048,1]
            X, Y = generate_linear_data(beta=beta,N = Ns[i],corr=0.0)
            nt, T , loss= nn_mse_exp(X,sd = 5,widths=widths,beta = beta, gen_nn=gen_nn)

            res[j,i] = loss
        nl.append(nt)


    to_save = 'nn_sim_res_2048_pop' + str(q) + '.txt'
    np.savetxt(to_save,res)
#np.savetxt('nn_sim_N_2048_pop.txt',nl)




