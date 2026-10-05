import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from Bay_fc_env import get_likelihood

nsimu = 600 # number of simulations

Lp = np.linspace(0,1,4) # list of reward probabilities
L = Lp

N = 4 # group size
dep = 0 # no depletion
ini = 0 # all agents initially in patch 0

# models. 0,0: non-interacting
LSI = [0,0,0,1,1] # 0: counting representation, 1: pulsatile representation
LIT = [0,1,2,1,2] # 1: threshold modulation, 2: belief modulation

for ip1 in range(len(Lp)):
        p1 = Lp[ip1]
        print('p1 =',p1)
        for ip0 in range(ip1+1):  
                p0 = Lp[ip0]
                print('p0 =',p0)
                for mo in range(len(LSI)):
                        SI = LSI[mo]
                        IT = LIT[mo]
                        print('Model',SI,IT)
                        Prob = get_likelihood(SI,IT,N,nsimu,p0,p1,dep,ini)
                        
                        #fname = './files/Sim2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(p1*10))+'_'+str(int(p0*10))+'_'+str(dep)+'_'+str(ini)+'.npy' 
                        #np.save(fname,Sim)
                        
                        fname = './files/Prob2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(p1*10))+'_'+str(int(p0*10))+'_'+str(dep)+'_'+str(ini)+'.npy' 
                        np.save(fname,Prob)

