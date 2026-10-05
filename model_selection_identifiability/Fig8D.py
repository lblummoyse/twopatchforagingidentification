import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from Bay_fc_sampling import get_simu_params
from matplotlib.ticker import LinearLocator

plt.rcParams.update({'font.size': 18,"text.usetex": True})

[_,_,_,_,tf,dt,_,_,_,nl] = get_simu_params()
names_m = ['NI','CT', 'CB','PT', 'PB']
LN = np.arange(2,7,1)
Lobs = np.arange(5,40,8)
Lratio = [0.2,0.2,0.2,0.8,0.2,0.2]
Ldep = np.arange(10,200,6)
Lin = np.arange(0,4+1,1)

methods = ['Bay'] # Bayesian inference

L = np.arange(1,11,2)/dt # list of sampling periods

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
                idrate = L[cond]
                qname = './files/conf_matrix_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'_'+str(idrate)+'.npy'                 
                Win = np.load(qname)

                fname = './files/RE_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'_'+str(idrate)+'.npy'                     
                ME = np.load(fname)
                print(ME.shape)
                Mm = np.zeros((5,nl))
                for mo in range(5):         
                        MWin[mo,cond] = Win[mo,mo]/np.sum(Win[mo,:])
                        m = 0
                        for i in range(5): # 5 parameters
                                m += np.count_nonzero(ME[mo,:,i] == 0)
                        m = m/(5*np.size(ME,axis=1))
                        M[mo,cond] = m  
        

        for mo in range(5): 
                ax[0].plot(L*dt,MWin[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='o') 
                ax[1].plot(L*dt,M[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='d')     
        ax[0].set_ylabel(r'$\mathcal{M}$')                       
        ax[1].set_ylabel(r'$\mathcal{P}$') 
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax[0].set_xticks(L*dt)       
        ax[0].set_xlabel(r'$\mathrm{Sampling \,\, period \, (a.u.)}$')    
        ax[1].set_xlabel(r'$\mathrm{Sampling \,\, period \, (a.u.)}$')   
        plt.tight_layout()
        title = './figures/conf_mRE_samp_'+str(method)+'.pdf'

        plt.savefig(title)  
        
plt.show()        
        
plt.show()                      





