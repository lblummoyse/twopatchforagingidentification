import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16,"text.usetex": True})
from Bay_fc_env import get_simu_params
from matplotlib.ticker import LinearLocator

[_,_,_,tf,dt,_,idrate,_,_,nl] = get_simu_params()
names_m = np.array(['$\mathrm{NI}$','$\mathrm{CT}$','$\mathrm{CB}$','$\mathrm{PT}$','$\mathrm{PB}$'])

Lp = np.linspace(0,1,4) # list of reward probabilities

methods = ['Bay'] # Bayesian inference

nsimu_obs = 20 # number of observations
N = 4 # group size
ratio = 0.2 # reward probability ratio p0/p1
dep = 0 # no depletion
ini = 0 # all agents initially in patch 0

MWin = np.zeros((5,len(Lp),len(Lp))) # 5 models, # conditions
M = np.zeros((5,len(Lp),len(Lp))) # 5 models, # conditions
for method in methods:
        for ip1 in range(len(Lp)):
                p1 = Lp[ip1]
                for ip0 in range(ip1+1):  
                        p0 = Lp[ip0]
                        print('p0',p0)
                        qname = './files/conf_matrix_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(p1*10))+'_'+str(int(p0*10))+'_dep_'+str(dep)+'_'+str(ini)+'.npy'                 
                        Win = np.load(qname)
                        print(Win)

                        fname = './files/RE_'+str(method)+'_'+str(N)+'_'+str(nsimu_obs)+'_'+str(int(p1*10))+'_'+str(int(p0*10))+'_'+str(dep)+'_'+str(ini)+'.npy'                     
                        ME = np.load(fname)
                        Mm = np.zeros((5,nl))
                        for mo in range(5): 
                                print('model',mo)
                                MWin[mo,ip1,ip0] = Win[mo,mo]/np.sum(Win[mo,:])  
                                m = 0
                                for i in range(5): # 5 parameters
                                        m += np.count_nonzero(ME[mo,:,i] == 0) 
                                        if ip0==0 and ip1==0:
                                                print('error',i,np.round(1-np.count_nonzero(ME[mo,:,i] == 0)/np.size(ME,axis=1),2))                         
                                        if ip0==0 and ip1==1:
                                                print('error bis',i,1-np.count_nonzero(ME[mo,:,i] == 0)/np.size(ME,axis=1))                                   
                                m = m/(5*np.size(ME,axis=1))
                                M[mo,ip1,ip0] = m  
        
        MWin_ = MWin.copy()
        M_ = M.copy()
     
        MWin[:,0,1:] = np.array([np.nan]*3)
        MWin[:,1,2:] = np.array([np.nan]*2)
        MWin[:,2,3:] = np.array([np.nan]*1)
        M[:,0,1:] = np.array([np.nan]*3)
        M[:,1,2:] = np.array([np.nan]*2)
        M[:,2,3:] = np.array([np.nan]*1)
        
        def transform(M):
                M = M.T
                n = M.shape[0]; diagonals = []
                for i in range(n):
                    diagonal = []
                    for j in range(n):
                        diagonal.append(M[(i + j) % n, j])
                    diagonals.append(diagonal)
                M2 = np.array(diagonals)
                return M2
        
        colors = ['black','darkred','red','goldenrod']
        Lal = [1,1,1,1]
        Lord = [3,0,1,2]
        
        fig, ax = plt.subplots(5,2,figsize=(9,12),sharex=False,sharey=False)
        for mo in range(5): 
                MWin2_ = transform(MWin[mo,:,:])
                M2_ = transform(M[mo,:,:])
                M2 = M2_.copy()
                MWin2 = MWin2_.copy()
                M2[1,:] = M2_[3,:]
                MWin2[1,:] = MWin2_[3,:]
                M2[3,:] = M2_[1,:]
                MWin2[3,:] = MWin2_[1,:]
                for ip1 in range(len(Lp)):
                        ax[mo,0].plot(Lp,MWin2[:,ip1],'o-',color=colors[ip1],alpha=Lal[ip1],label=rf'$p^1 = {np.round(Lp[ip1],2)}$',zorder=Lord[ip1])
                        ax[mo,1].plot(Lp,M2[:,ip1],'d-',color=colors[ip1],alpha=Lal[ip1],label=r'$p^1 = $ '+str(np.round(Lp[ip1],2)),zorder=Lord[ip1])
                ax[mo,0].set_xlabel(r'$p^1-p^0$')
                ax[mo,1].set_xlabel(r'$p^1-p^0$')
                ax[mo,0].set_xticks(np.round(Lp,2))
                ax[mo,1].set_xticks(np.round(Lp,2))
                ax[mo,0].set_ylabel(r'$\mathcal{M}$')
                
                ax[mo,0].set_title(names_m[mo],x=1.1)
                ax[mo,1].set_ylabel(r'$\mathcal{P}$')
                ax[mo,0].set_ylim(0.77,1)
                ax[mo,1].set_ylim(0.675,0.965)
                ax[mo,0].yaxis.set_major_locator(plt.MaxNLocator(3))
                ax[mo,1].yaxis.set_major_locator(plt.MaxNLocator(3))
        ax[0,0].legend(loc="upper center",bbox_to_anchor=(1.1, 2),ncols=5,frameon=False,handletextpad=0.5,columnspacing=1.2) 
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.25,hspace=0.7)
        title = './figures/conf_mRE_env_'+str(method)+'.pdf'
        fig.savefig(title)
        
 
plt.show()        
        
 
