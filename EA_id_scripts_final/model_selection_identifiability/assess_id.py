import sys
sys.path.insert(0, '../functions') 

import numpy as np
import matplotlib.pyplot as plt
from simu_fc import Simu
from Bay_fc import get_simu_params, get_P, model_samples, get_posterior, get_distance
plt.rcParams.update({'font.size':18})

[_,_,_,_,tf,dt,_,idrate,_,_,nl] = get_simu_params()
new = 1 # 0 not new, 1 new 
if new==0:
        print('not new')
elif new>0:
        print('new!')
        
nsimu = 600 # number of simulations
n_attempts = 400 # number of attempts

# different varying experimental conditions
a = 2 # 0: obs, 1: N, 2: dep, 3: ini, 4: single matrix
print('a =',a)
if a==0:
        Lobs = np.arange(1,20+5,5) # number of observations
        L = Lobs
        Lin = np.zeros(len(L)) # number of agents initially in the patch 1
        LN = 4*np.ones(len(L)) # group size
        Ldep = np.zeros(len(L)) # maximum amount of food in a patch, if depletion
        Lratio = 0.2*np.ones(len(L)) # reward probability ratio p0/p1
elif a==1:
        LN = np.arange(2,20,4)
        L = LN
        Ldep = np.zeros(len(L))
        Lin = np.zeros(len(L))
        Lratio = 0.2*np.ones(len(L))
        Lobs = 20*np.ones(len(L))
elif a==2:
        Ldep = np.linspace(10,150,5)
        L = Ldep
        LN = 4*np.ones(len(L))
        Lin = np.zeros(len(L))
        Lratio = 0.2*np.ones(len(L))
        Lobs = 20*np.ones(len(L))
elif a==3:
        Lin = np.arange(0,4+1,1)
        L = Lin
        Ldep = np.zeros(len(L))
        LN = 4*np.ones(len(L))
        Lratio = 0.2*np.ones(len(L))
        Lobs = 20*np.ones(len(L))
elif a==4:
        Lin = [0]
        Ldep = [0]
        LN = [4]
        Lratio = [0.2]
        Lobs = [20]
        L = Lobs

print('L =',L)

# models. 0,0: non-interacting
LSI = [0,0,0,1,1] # 0: counting representation, 1: pulsatile representation
LIT = [0,1,2,1,2] # 1: threshold modulation, 2: belief modulation

methods = ['Bay','dis'] # Bayesian inference or distance minimization
im = 0 
method = methods[im]
print(method)
for cond in range(len(L)):
        print('condition', cond)
        N = int(LN[cond])
        nsimu_obs = int(Lobs[cond])
        ratio = Lratio[cond]
        dep = int(Ldep[cond])
        ini = int(Lin[cond])
        if im==0:
                ML = np.zeros((5,nl,nl,nl,nl,nl,1,int(tf/(idrate*dt)),N+1))
        elif im==1:
                Sim = np.zeros((5,nl,nl,nl,nl,nl,1,nsimu,int(tf/(idrate*dt))))
        for mo in range(len(LSI)):
                SI_ = LSI[mo]
                IT_ = LIT[mo]         
                if im==0:
                        fname = './files/Prob2_'+str(SI_)+'_'+str(IT_)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy'
                        ML[mo] = np.load(fname) # likelihood of each model
                        np.save(fname,ML[mo])
                elif im==1:
                        fname = './files/Sim2_'+str(SI_)+'_'+str(IT_)+'_'+str(N)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy' 
                        Sim[mo] = np.load(fname) # simu of each model

        Win = np.zeros((5,5))
        ME = np.zeros((5,n_attempts,5)) # 5 models, n attempts, 5 parameters               
        for mo_ in range(len(LSI)):
                SI_ = LSI[mo_]
                IT_ = LIT[mo_]                                        
                print('True model',SI_,IT_)
                for na in range(n_attempts):
                        [i_,j_,l_,k_,c_] = np.random.randint(0,nl,size = 5) # arg true
                        arg_true = np.array([i_,j_,l_,k_,c_])
                        P_true = get_P(i_,j_,l_,k_,c_)
                        obs = model_samples(IT_,SI_,N,ratio,dep,ini,nsimu_obs,*P_true)
                        
                        MP = np.zeros(5)
                        for mo in range(len(LSI)):
                                SI = LSI[mo]
                                IT = LIT[mo] 
                                if im==0:
                                        Post = get_posterior(ML[mo],obs,N,nl,nsimu_obs)
                                        MP[mo] = np.max(Post) # posterior
                                else:
                                        D = get_distance(Sim[mo],obs,N,nl,nsimu_obs)
                                        MP[mo] = np.min(D)
                                if SI==SI_ and IT==IT_: # compute grid error                                                
                                        if im==0:
                                                arg = np.unravel_index(np.argmax(Post, axis=None), Post.shape)
                                        elif im==1:
                                                arg = np.unravel_index(np.argmin(D, axis=None), D.shape)
                                        E = np.abs(arg - arg_true)
                                        ME[mo,na,:] = E
                        if im==0:
                                win = np.argmax(MP)
                        elif im==1:
                                win = np.argmin(MP)
                        Win[mo_,win] += 1
        fname = './files/conf_matrix_'+method+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.npy'
        if new==0:
                MWin = np.load(fname)
                MWin = MWin + Win    
                np.save(fname, MWin) 
                Win = MWin   
                Win = Win/np.sum(Win[0,:])  
        elif new>0:
                np.save(fname, Win)
                Win = Win/n_attempts
        
        fig,ax = plt.subplots(1,1)
        imax = ax.imshow(Win,vmin=0,vmax=1)
        models_name = ['NI','CT', 'CB','PT', 'PB']
        ax.set_xticks(range(5), labels=models_name,rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(5), labels=models_name)
        for i in range(5):
            for j in range(5):
                if i==j:
                        color = 'k'
                else:
                        color = 'w'
                text = ax.text(j, i, np.round(Win[i, j],2),ha='center', va='center', color=color)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')  
        plt.tight_layout()
        title = './figures/Conf_'+method+'_'+str(N)+'_nsimu_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.pdf'
        if a==4:
                fig.savefig(title)

        ####### Error plot #########
        ME[0,:,3] = np.zeros(n_attempts) # non-interacting model, no error associated with kappa
        fname = './files/RE_'+method+'_'+str(N)+'_'+str(nsimu_obs)+'_'+str(int(ratio*10))+'_'+str(dep)+'_'+str(ini)+'.npy'                     
        if new==0:
                ME2 = np.load(fname)
                ME2 = np.concatenate([ME2,ME],axis=1)  
                np.save(fname, ME2)
                ME = ME2
        elif new>0:
                np.save(fname, ME)

plt.show()                      





