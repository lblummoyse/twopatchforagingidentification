import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18,"text.usetex": True})
from Bay_fc import get_simu_params

nl = get_simu_params()[-1]
names_m = ['NI','CT', 'CB','PT', 'PB']

LN = np.arange(2,20,4)
Lobs = np.arange(1,20+5,5)
Lratio = [0.2,0.2,0.2,0.8,0.2,0.2]
Ldep = np.linspace(10,150,5)
Lin = np.arange(0,4+1,1)

methods = ['Bay'] #['dis']

# varying experimental condition
a = 2 # 0: obs, 1: N, 2: dep, 3: ini
if a==1:
        L = LN
elif a==2:
        L = Ldep
elif a==3:
        L = Lin
elif a==0:
        L = Lobs
        
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
                if a==1:
                        N = L[cond]
                elif a==2:
                        dep = int(L[cond])
                elif a==3:
                        ini = L[cond]
                elif a==0:
                        nsimu_obs = L[cond]
                qname = './files/conf_matrix_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.npy'                 
                Win = np.load(qname)
                fname = './files/RE_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy'                     
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
                if a==3:
                        ax[0].plot(L/4,MWin[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='o') 
                        ax[1].plot(L/4,M[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='d')    
                else:
                        ax[0].plot(L,MWin[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='o') 
                        ax[1].plot(L,M[mo,:],color=colors[mo],label=names_m[mo],ls=lstyles[mo],marker='d')     
        ax[0].set_ylabel(r'$\mathcal{M}$')                       
        ax[1].set_ylabel(r'$\mathcal{P}$') 
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
        if a==1:
                ax[0].set_xlabel(r'$N$')                       
                ax[1].set_xlabel(r'$N$') 
                ax[0].set_xticks(L) 
        elif a==3:    
                ax[0].set_xlabel(r'$Q_\mathrm{in}$')    
                ax[1].set_xlabel(r'$Q_\mathrm{in}$') 
                ax[0].set_xticks(L/4) 
        elif a==0:
                ax[0].set_xlabel(r'$\# \, \mathrm{observations}$')   
                ax[1].set_xlabel(r'$\# \, \mathrm{observations}$')  
                ax[0].set_xticks(L)
        elif a==2:
                ax[0].set_xlabel(r'$A_m$')    
                ax[1].set_xlabel(r'$A_m$')  
                ax[0].set_xticks(L) 
        plt.tight_layout()
        if a==1:
                title = './figures/conf_mRE_N_'+str(method)+'.pdf' 
        elif a==3:
                title = './figures/conf_mRE_ini_'+str(method)+'.pdf'
        elif a==2:
                title = './figures/conf_mRE_dep_'+str(method)+'.pdf' 
        elif a==0:
                title = './figures/conf_mRE_obs_'+str(method)+'.pdf'
        plt.savefig(title)        
plt.show()        
        
        

