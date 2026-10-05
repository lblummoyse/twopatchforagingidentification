import sys
sys.path.insert(0, '../functions') 
sys.path.insert(0, '../model_selection_identifiability')  

import matplotlib.pyplot as plt
import numpy as np
from simu_fc import Simu
from Bay_fc import get_boundaries
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size': 18,"text.usetex": True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

B = 0.1 # diffusion coefficient
eta = 1/2 # counting parameter
dep = 0 # 0: no depletion, 1: depletion 
Lco2 = [0,1] #0: counting, 1: pulsatile
N = 4 # group size
Lco = [1,2] #0 no coupling, 1 threshold, 2 belief
Lp = np.linspace(0,1,4) # list of reward probabilities
Am = 0 # maximum amount of food (0 here because no depletion)
tf = 50 # duration of the simulation
dt = 0.1  # time step
iDeltar = 1 # >1 if time step between resource distribution times, =1 otherwise 
idrate = int(1/dt) # sampling period
Ttr = 0.2 # travel time
Lt = np.arange(0,tf,dt) # list of times
nsimu = 50 # number of simulations

nl = 1 # number of values per parameter
[thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
Lth = np.linspace(thmin,thmax,nl)
La = np.linspace(amin,amax,nl)
Ls = np.linspace(smin,smax,nl)
Lz = np.linspace(zmin,zmax,nl)
Lk = np.linspace(kmin,kmax,nl)

# Initial accuracy
Qin = np.zeros((nsimu,N))

# Recording
LRTs = np.zeros((len(Lp),len(Lp)))
LQs = np.zeros((len(Lp),len(Lp)))
Lcohs = np.zeros((len(Lp),len(Lp)))
MRT = np.zeros((2,2,len(Lp),len(Lp)))
MQ = np.zeros((2,2,len(Lp),len(Lp)))
Mcoh = np.zeros((2,2,len(Lp),len(Lp)))

for theta in Lth:
        print('theta=',theta)
        for alpha in La:
                for zeta in Lz:
                        for s in Ls:
                                for ip1 in range(len(Lp)):
                                        p1 = Lp[ip1] 
                                        for ip0 in range(ip1+1):  
                                                p0 = Lp[ip0]
                                                [RTs,LaccS,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,0,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                LRTs[ip0,ip1] += np.mean(RTs)
                                                LQs[ip0,ip1] += np.mean(Macc)
                                                Lcohs[ip0,ip1] += np.mean(np.abs(np.diff(Macc,axis=1)))
                                                for co2 in Lco2:        
                                                        for co in Lco:
                                                                for kd in Lk:

                                                                        [RT,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                                        if len(RT)==0:
                                                                                RT = np.array([tf])
                                                                        MRT[co2,co-1,ip0,ip1] += np.mean(RT)
                                                                        MQ[co2,co-1,ip0,ip1] += np.mean(Macc)
                                                                        Mcoh[co2,co-1,ip0,ip1] += np.mean(np.abs(np.diff(Macc,axis=1)))


LRTs = LRTs/nl**4
LQs = LQs/nl**4
Lcohs = Lcohs/nl**4
MRT = MRT/nl**5
MQ = MQ/nl**5
Mcoh = Mcoh/nl**5

np.save('./files/LRTs_env',LRTs)
np.save('./files/LQs_env',LQs)
np.save('./files/Lcohs_v_env',Lcohs)
np.save('./files/MRT_env',MRT)
np.save('./files/MQ_env',MQ)
np.save('./files/Mcoh_v_env',Mcoh)


'''LRTs = np.load('./files/LRTs_env.npy')
LQs = np.load('./files/LQs_env.npy')
Lcohs = np.load('./files/Lcohs_v_env.npy')
MRT = np.load('./files/MRT_env.npy')
MQ = np.load('./files/MQ_env.npy')
Mcoh = np.load('./files/Mcoh_v_env.npy')'''

names = np.array([['$\mathrm{CT}$','$\mathrm{CB}$'],['$\mathrm{PT}$','$\mathrm{PB}$']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])

lstyles = ['-','--']

lw = 3

min0 = np.min([np.min(LRTs[LRTs>0]),np.min(MRT[MRT>0])])
max0 = np.max([np.max(LRTs),np.max(MRT)])
min1 = np.min([np.min(LQs[LQs>0]),np.min(MQ[MQ>0])])
max1 = np.max([np.max(LQs),np.max(MQ)])
min2 = np.min([np.min(Lcohs[Lcohs>0]),np.min(Mcoh[Mcoh>0])])/(idrate*dt)
max2 = np.max([np.max(Lcohs),np.max(Mcoh)])/(idrate*dt)

LRTs[1:,0] = np.array([np.nan]*3)
LRTs[2:,1] = np.array([np.nan]*2)
LRTs[3:,2] = np.array([np.nan]*1)
MRT[:,:,1:,0] = np.array([np.nan]*3)
MRT[:,:,2:,1] = np.array([np.nan]*2)
MRT[:,:,3:,2] = np.array([np.nan]*1)

LQs[1:,0] = np.array([np.nan]*3)
LQs[2:,1] = np.array([np.nan]*2)
LQs[3:,2] = np.array([np.nan]*1)
MQ[:,:,1:,0] = np.array([np.nan]*3)
MQ[:,:,2:,1] = np.array([np.nan]*2)
MQ[:,:,3:,2] = np.array([np.nan]*1)

Lcohs[1:,0] = np.array([np.nan]*3)
Lcohs[2:,1] = np.array([np.nan]*2)
Lcohs[3:,2] = np.array([np.nan]*1)
Mcoh[:,:,1:,0] = np.array([np.nan]*3)
Mcoh[:,:,2:,1] = np.array([np.nan]*2)
Mcoh[:,:,3:,2] = np.array([np.nan]*1)

def transform(M):
        n = M.shape[0]; diagonals = []
        for i in range(n):
            diagonal = []
            for j in range(n):
                diagonal.append(M[(i + j) % n, j])
            diagonals.append(diagonal)
        M2_ = np.array(diagonals)
        M2 = M2_.copy()
        M2[1,:] = M2_[3,:]
        M2[3,:] = M2_[1,:]
        return M2

colors = ['black','darkred','red','goldenrod']
Lord = [3,0,1,2]
fig, ax = plt.subplots(5,3,figsize=(12,14),sharex=False,sharey=False)
LRTs2 = transform(LRTs)
LQs2 = transform(LQs)
Lcohs2 = transform(Lcohs)
for ip1 in range(len(Lp)):
        ax[0,2].plot(Lp,Lcohs2[:,ip1],'o-',color=colors[ip1],zorder=Lord[ip1])
ax[0,0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0,1].set_ylabel(r'$Q$')
ax[0,2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
ax[0,0].set_xlabel(r'$p^1-p^0$')
ax[0,1].set_xlabel(r'$p^1-p^0$')
ax[0,2].set_xlabel(r'$p^1-p^0$')
ax[0,0].set_xticks(np.round(Lp,2))
ax[0,1].set_xticks(np.round(Lp,2))
ax[0,2].set_xticks(np.round(Lp,2))
ax[0,1].set_title('$\mathrm{NI}$')
mo = 0
ax[mo,2].set_ylim(0.03,0.23)
ax[mo,2].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[mo,2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)

for co2 in Lco2:        
        for co in Lco:
                mo = co2*2+co
                MRT2 = transform(MRT[co2,co-1,:,:])
                MQ2 = transform(MQ[co2,co-1,:,:])
                Mcoh2 = transform(Mcoh[co2,co-1,:,:])
                for ip1 in range(len(Lp)):
                        ax[mo,2].plot(Lp,Mcoh2[:,ip1],'o-',color=colors[ip1],zorder=Lord[ip1])
                ax[mo,0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
                ax[mo,1].set_ylabel(r'$Q$')
                ax[mo,1].set_title(names[co2,co-1])
                ax[mo,2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
                ax[mo,0].set_xlabel(r'$p^1-p^0$')
                ax[mo,1].set_xlabel(r'$p^1-p^0$')
                ax[mo,2].set_xlabel(r'$p^1-p^0$')
                ax[mo,0].set_xticks(np.round(Lp,2))
                ax[mo,1].set_xticks(np.round(Lp,2))
                ax[mo,2].set_xticks(np.round(Lp,2))
                ax[mo,2].set_ylim(0.03,0.23)
                ax[mo,2].yaxis.set_major_locator(plt.MaxNLocator(3))
                ax[mo,2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)

ax[0,1].legend(loc="upper center",ncols=5,bbox_to_anchor=(0.5, 2.5),frameon=False,handletextpad=0.5,columnspacing=1.5) 
plt.subplots_adjust(hspace=1.5, wspace=0.45)
title = './figures/Metrics_env_group.pdf'
fig.savefig(title,bbox_inches='tight')
plt.show()

