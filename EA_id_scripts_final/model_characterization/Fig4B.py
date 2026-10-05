import sys
sys.path.insert(0, '../functions') 
sys.path.insert(0, '../model_selection_identifiability')  

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from simu_fc import Simu
from Bay_fc import get_boundaries
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size': 24,"text.usetex": True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


B = 0.1 # diffusion coefficient
eta = 1/2 # counting parameter
dep = 0 # no depletion
Lco2 = [0,1] # 0: counting, 1: pulsatile
Lco = [1,2] #0 no coupling, 1 threshold, 2 belief

N = 4 # group size
Lp = np.arange(0,N+1,1) # initial repartition of agents

Am = 0 # maximum amount of food (=0 if no depletion)
tf = 50 # duration of the simulation
dt = 0.1 # time step
iDeltar = 1 # >1 if time step between resource distribution times, =1 otherwise 
idrate = int(1/dt) # sampling period
Ttr = 0.2 # travel time
Lt = np.arange(0,tf,dt) # list of times
nsimu = 50 # number of simulations
p1 = 0.8; ratio = 0.2; p0 = p1*ratio # reward probabilities

nl = 3 # number of values per parameter
[thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
Lth = np.linspace(thmin,thmax,nl)
La = np.linspace(amin,amax,nl)
Ls = np.linspace(smin,smax,nl)
Lz = np.linspace(zmin,zmax,nl)
Lk = np.linspace(kmin,kmax,nl)

LRTs = np.zeros(len(Lp))
LQs = np.zeros(len(Lp))
Lcohs = np.zeros(len(Lp))
Lcohs_v = np.zeros(len(Lp))
MRT = np.zeros((2,2,len(Lp)))
MQ = np.zeros((2,2,len(Lp)))
Mcoh = np.zeros((2,2,len(Lp)))
Mcoh_v = np.zeros((2,2,len(Lp)))

for theta in Lth:
        print('theta=',theta)
        for alpha in La:
                for zeta in Lz:
                        for s in Ls:
                                for ip in range(len(Lp)):
                                        prop = int(Lp[ip])   
                                        Qin = np.zeros((nsimu,N))
                                        Qin[:,:prop] = np.ones((nsimu,prop))
                                        [RTs,LaccS,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,0,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                        LRTs[ip] += np.mean(RTs)
                                        LQs[ip] += np.mean(Macc)
                                        Lcohs[ip] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0)))) 
                                        Lcohs_v[ip] += np.mean(np.abs(np.diff(Macc,axis=1)))
                                        for co2 in Lco2:        
                                                for co in Lco:
                                                        for kd in Lk:
                                                                [RT,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                                if len(RT)==0:
                                                                        RT = np.array([tf])                                                              
                                                                MRT[co2,co-1,ip] += np.mean(RT)
                                                                MQ[co2,co-1,ip] += np.mean(Macc)
                                                                Mcoh[co2,co-1,ip] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0))))
                                                                Mcoh_v[co2,co-1,ip] += np.mean(np.abs(np.diff(Macc,axis=1)))
                                                                
LRTs = LRTs/nl**4
LQs = LQs/nl**4
Lcohs = Lcohs/nl**4
Lcohs_v = Lcohs_v/nl**4
MRT = MRT/nl**5
MQ = MQ/nl**5
Mcoh = Mcoh/nl**5
Mcoh_v = Mcoh_v/nl**5

np.save('./files/LRTs_ini',LRTs)
np.save('./files/LQs_ini',LQs)
np.save('./files/Lcohs_ini',Lcohs)
np.save('./files/Lcohs_v_ini',Lcohs_v)
np.save('./files/MRT_ini',MRT)
np.save('./files/MQ_ini',MQ)
np.save('./files/Mcoh_ini',Mcoh)
np.save('./files/Mcoh_v_ini',Mcoh_v)
'''
LRTs = np.load('./files/LRTs_ini.npy')
LQs = np.load('./files/LQs_ini.npy')
Lcohs = np.load('./files/Lcohs_ini.npy')
Lcohs_v = np.load('./files/Lcohs_v_ini.npy')
MRT = np.load('./files/MRT_ini.npy')
MQ = np.load('./files/MQ_ini.npy')
Mcoh = np.load('./files/Mcoh_ini.npy')
Mcoh_v = np.load('./files/Mcoh_v_ini.npy')'''
                                                        
names = np.array([['CT','CB'],['PT','PB']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])
lstyles = ['-','--']
lw = 3
# Fig4B
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
ax[0].set_xticks([0,0.25,0.5,0.75,1])
ax[0].plot(Lp/N,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Lp/N,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$Q_\text{in}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Lp/N,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Lp/N,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$Q_\text{in}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Lp/N,Mcoh[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Lp/N,Lcohs/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$Q_\text{in}$')        
plt.tight_layout()
title = './figures/Metrics_ini.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 

#########################################################
# SFig2E
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)
ax[0].set_xticks([0,0.25,0.5,0.75,1])
ax[0].plot(Lp/N,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Lp/N,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$Q_\text{in}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Lp/N,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Lp/N,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$Q_\text{in}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Lp/N,Mcoh_v[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Lp/N,Lcohs_v/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$Q_\text{in}$')        
plt.tight_layout()
title = './figures/Metrics_ini_group.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 
              
plt.show()

