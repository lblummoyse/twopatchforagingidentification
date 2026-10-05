import sys
sys.path.insert(0, '../functions') 
sys.path.insert(0, '../model_selection_identifiability')  

import matplotlib.pyplot as plt
import numpy as np
from simu_fc import Simu
from Bay_fc import get_boundaries
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size': 24,'text.usetex': True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

B = 0.1 # diffusion coefficient
eta = 1/2 # counting parameter
dep = 0 # no depletion
Lco2 = [0,1] #0: counting, 1: pulsatile
Lco = [1,2] #0 no coupling, 1 threshold, 2 belief
N = 4 # group size

nl = 3 # numer of values per parameter
[thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
Lth = np.linspace(thmin,thmax,nl)
La = np.linspace(amin,amax,nl)
Ls = np.linspace(smin,smax,nl)
Lz = np.linspace(zmin,zmax,nl)
Lkd = np.linspace(0,0.9,6)

Am = 0 # maximum amount of food (=0 if no depletion)
tf = 50 # duration of the simulation
dt = 0.1 # time step
iDeltar = 1 # >1 if time step between resource distribution times, =1 otherwise 
idrate = int(1/dt) # sampling period 
Ttr = 0.2 # travel time
Lt = np.arange(0,tf,dt) # list of times
nsimu = 50  # number of simulations
p1 = 0.8; ratio = 0.2; p0 = p1*ratio # reward probabilities

Qin = np.zeros((nsimu,N)) # initial repartition of agents

# Recording
RTs = 0; Qs = 0; cohs = 0; cohs_v=0
MRT = np.zeros((2,2,len(Lkd)))
MQ = np.zeros((2,2,len(Lkd)))
Mcoh = np.zeros((2,2,len(Lkd)))
MRT_v = np.zeros((2,2,len(Lkd)))
MQ_v = np.zeros((2,2,len(Lkd)))
Mcoh_v = np.zeros((2,2,len(Lkd)))


for theta in Lth:
        print('theta=',theta)
        for alpha in La:
                for zeta in Lz:
                        for s in Ls:
                                [RTs_,LaccS,Maccs_] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,0,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                RTs += np.mean(RTs_)
                                Qs += np.mean(Maccs_)
                                cohs += np.mean(np.abs(np.diff(np.mean(Maccs_,axis=0))))
                                cohs_v += np.mean(np.abs(np.diff(Maccs_,axis=1)))
                                for co2 in Lco2:        
                                        for co in Lco:
                                                for ikd in range(len(Lkd)):
                                                        kd = Lkd[ikd]    
                                                        [RT,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                        if len(RT)==0:
                                                                        RT = np.array([tf])
                                                        MRT[co2,co-1,ikd] += np.mean(RT)
                                                        MQ[co2,co-1,ikd] += np.mean(Macc)
                                                        c_group = np.abs(np.diff(Macc,axis=1))
                                                        c = np.abs(np.diff(np.mean(Macc,axis=0)))
                                                        Mcoh[co2,co-1,ikd] += np.mean(c) # mean over time
                                                        Mcoh_v[co2,co-1,ikd] += np.mean(c_group) # mean over time

RTs = RTs/nl**4
Qs = Qs/nl**4
cohs = cohs/nl**4
cohs_v = cohs_v/nl**4
MRT = MRT/nl**4
MQ = MQ/nl**4
Mcoh = Mcoh/nl**4
Mcoh_v = Mcoh_v/nl**4


np.save('./files/RTs_k',RTs)
np.save('./files/Qs_k',Qs)
np.save('./files/cohs_k',cohs)
np.save('./files/cohs_v_k',cohs_v)
np.save('./files/MRT_k',MRT)
np.save('./files/MQ_k',MQ)
np.save('./files/Mcoh_k',Mcoh)
np.save('./files/Mcoh_v_k',Mcoh_v)
'''
RTs = np.load('./files/RTs_k.npy')
Qs = np.load('./files/Qs_k.npy')
cohs = np.load('./files/cohs_k.npy')
cohs_v = np.load('./files/cohs_v_k.npy')
MRT = np.load('./files/MRT_k.npy')
MQ = np.load('./files/MQ_k.npy')
Mcoh = np.load('./files/Mcoh_k.npy')
Mcoh_v = np.load('./files/Mcoh_v_k.npy')'''

names = np.array([['CT','CB'],['PT','PB']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])
lstyles = ['-','--']
lw = 3
al = 0.2
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(2))
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
ax[0].axhline(RTs,c='black',ls='-',label='NI')
for co2 in Lco2:        
        for co in Lco:
                m = MRT[co2,co-1,:]; s = MRT_v[co2,co-1,:]
                ax[0].plot(Lkd,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\overline{\kappa}$') 
ax[1].axhline(Qs,c='black',ls='-')
for co2 in Lco2:        
        for co in Lco:
                m = MQ[co2,co-1,:]; s = MQ_v[co2,co-1,:]
                ax[1].plot(Lkd,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\overline{\kappa}$')        
ax[2].axhline(cohs/(idrate*dt),c='black',ls='-')
for co2 in Lco2:        
        for co in Lco:
                m = Mcoh[co2,co-1,:]; s = Mcoh_v[co2,co-1,:]
                ax[2].plot(Lkd,Mcoh[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].set_ylabel(r'$D \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\overline{\kappa}$')   
plt.tight_layout()
title = './figures/Metrics_k.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 


#######################################################
# SFig2A
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(3))
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)
ax[0].axhline(RTs,c='black',ls='-',label='NI')
for co2 in Lco2:        
        for co in Lco:
                m = MRT[co2,co-1,:]; s = MRT_v[co2,co-1,:]
                ax[0].plot(Lkd,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\overline{\kappa}$') 
ax[1].axhline(Qs,c='black',ls='-')
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Lkd,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\overline{\kappa}$')        
ax[2].axhline(cohs_v/(idrate*dt),c='black',ls='-')
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Lkd,Mcoh_v[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\overline{\kappa}$')   

plt.tight_layout()
title = './figures/Metrics_k_group.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 
              
plt.show()

