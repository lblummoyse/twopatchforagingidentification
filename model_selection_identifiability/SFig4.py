import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from simu_fc import Simu
plt.rcParams.update({'font.size': 18,"text.usetex": True})
from Bay_fc_nlfc import get_simu_params
from matplotlib.ticker import LinearLocator

[_,_,_,_,tf,dt,_,idrate,_,_] = get_simu_params()
names_m = ['NI','CT', 'CB','PT', 'PB']

methods = ['Bay'] # Bayesian inference

L = np.arange(2,5,1) # list of nl: number of values per parameter
        
# default experimental conditions        
nsimu_obs = 20 # number of observations
N = 4 # group size
ratio = 0.2 # reward probability ratio p0/p1
dep = 0 # no depletion
ini = 0 # all agents initially in the patch 0

colors = ['k',"#0072B2", "#56B4E9","#D55E00","#E69F00"]
lstyles = ['-','-','--','-','--']
MWin = np.empty((5,len(L))) # 5 models, # conditions
M = np.empty((5,len(L))) # 5 models, # conditions

for method in methods:
        fig, ax = plt.subplots(1,2,figsize=(10,3.5),sharex=True,sharey=False)
        for cond in range(len(L)):
                nl = L[cond]
                qname = './files/conf_matrix_nlfc_'+str(nl)+'.npy'
                Win = np.load(qname)
                fname =  './files/RE_nlfc_'+str(nl)+'.npy'
                ME = np.load(fname)
                Mm = np.zeros((5,nl))
                for mo in range(5): 
                        MWin[mo,cond] = Win[mo,mo]/np.sum(Win[mo,:])
                        m = 0
                        for i in range(5): # 5 parameters
                                m += np.count_nonzero(ME[mo,:,i] == 0)
                                        
                        m = m/(5*np.size(ME,axis=1))
                        M[mo,cond] = m  
        for mo in range(5): 
                ax[0].plot(L,MWin[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='o')
                ax[1].plot(L,M[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='d')     
        ax[0].set_ylabel(r'$\mathcal{M}$')                       
        ax[1].set_ylabel(r'$\mathcal{P}$') 
        ax[0].set_xlabel(r'$n_l$')    
        ax[1].set_xlabel(r'$n_l$') 
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(4)) 
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(4)) 
        ax[0].set_xticks(L) 
        plt.tight_layout()
        
        title = './figures/conf_mRE_nl.pdf'
        plt.savefig(title)  
        
plt.show()        
        

