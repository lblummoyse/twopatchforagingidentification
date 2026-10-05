import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from Bay_fc_nlfc import get_likelihood

nsimu = 600 # number of simulations

N = 4 # group size
ini = 0 # initial accuracy (all in patch 0)
dep = 0 # no depletion
ratio = 0.2 # reward probability ratio p0/p1
 
L = np.arange(2,5,1) # list of nl: number of values per parameter

# models. 0,0: non-interacting
LSI = [0,0,0,1,1] # 0: counting representation, 1: pulsatile representation
LIT = [0,1,2,1,2] # 1: threshold modulation, 2: belief modulation

for cond in range(len(L)):
        nl = L[cond]
        print(nl)
        for mo in range(len(LSI)):
                SI = LSI[mo]
                IT = LIT[mo]
                print('Model',SI,IT)
                Prob = get_likelihood(nl,SI,IT,N,nsimu,ratio,dep,ini)

                fname = './files/Prob_nlfc_'+str(SI)+'_'+str(IT)+'_'+str(nl)+'.npy' 
                np.save(fname,Prob)

