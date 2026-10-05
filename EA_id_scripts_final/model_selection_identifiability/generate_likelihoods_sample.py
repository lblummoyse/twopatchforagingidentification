import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from Bay_fc_sampling import get_likelihood, get_simu_params

nsimu = 600 # number of simulations
[_,_,_,_,_,dt,_,_,_,_] = get_simu_params()

N = 4 # group size
ini = 0 # initial accuracy (all in patch 0)
dep = 0 # no depletion
ratio = 0.2 # reward probability ratio p0/p1

L = np.arange(1,11,2)/dt

# models. 0,0: non-interacting
LSI = [0,0,0,1,1] # 0: counting representation, 1: pulsatile representation
LIT = [0,1,2,1,2] # 1: threshold modulation, 2: belief modulation

for cond in range(len(L)):
        idrate = L[cond]
        print(idrate)
        for mo in range(len(LSI)):
                SI = LSI[mo]
                IT = LIT[mo]
                print('Model',SI,IT)
                Sim, Prob = get_likelihood(SI,IT,N,nsimu,ratio,dep,ini,idrate)
                
                fname = './files/Sim2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'_'+str(idrate)+'.npy' 
                np.save(fname,Sim)
                
                fname = './files/Prob2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'_'+str(idrate)+'.npy' 
                np.save(fname,Prob)

