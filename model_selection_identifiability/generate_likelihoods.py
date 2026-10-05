import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from Bay_fc import get_likelihood

nsimu = 1#600 # number of simulations

# varying experimental condition
a = 2 # 1: N, 2: dep, 3: ini, 4: single matrix
print('a=',a)
if a==1:
        LN = np.arange(2,22,4)
        L = LN
        Ldep = np.zeros(len(L))
        Lin = np.zeros(len(L))
        Lratio = 0.2*np.ones(len(L))
elif a==2:
        Ldep = np.linspace(10,150,5)
        L = Ldep
        LN = 4*np.ones(len(L))
        Lin = np.zeros(len(L))
        Lratio = 0.2*np.ones(len(L))
elif a==3:
        Lin = np.arange(0,4+1,1)
        L = Lin
        Ldep = np.zeros(len(L))
        LN = 4*np.ones(len(L))
        Lratio = 0.2*np.ones(len(L))
elif a==4:    
        LN = [4]
        Ldep = [0]
        Lratio = [0.2]
        Lin = [0]  
        L = Lin   

# models. 0,0: non-interacting
LSI = [0,0,0,1,1] # 0: counting representation, 1: pulsatile representation
LIT = [0,1,2,1,2] # 1: threshold modulation, 2: belief modulation

for cond in range(len(L)):
        print('cond',cond)
        N = int(LN[cond])
        ratio = Lratio[cond]
        dep = int(Ldep[cond])
        ini = int(Lin[cond])

        for mo in range(len(LSI)):
                SI = LSI[mo]
                IT = LIT[mo]
                print('Model',SI,IT)
                Sim, Prob = get_likelihood(SI,IT,N,nsimu,ratio,dep,ini)
                
                fname = './files/Sim2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy' 
                np.save(fname,Sim)
                
                fname = './files/Prob2_'+str(SI)+'_'+str(IT)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy' 
                np.save(fname,Prob)

