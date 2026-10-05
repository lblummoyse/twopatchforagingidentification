import sys
sys.path.insert(0, '../functions') 
sys.path.insert(0, '../model_selection_identifiability')  

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from simu_fc import Simu
from Bay_fc import get_boundaries
plt.rcParams.update({'font.size': 24,"text.usetex": True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LinearLocator

B = 0.1 # diffusion coefficient
eta = 1/2 # counting parameter
dep = 0 # no depletion
Lco2 = [0,1] #0: counting, 1: pulsatile
Lco = [1,2] #0 no coupling, 1 threshold, 2 belief
N = 4 # group size

nl = 3 # number of values per parameter
[thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
Lth = np.linspace(thmin,thmax,nl)
La = np.linspace(amin,amax,nl)
Lk = np.linspace(kmin,kmax,nl)
Lz = np.linspace(zmin,zmax,nl)
Ls = np.linspace(0,0.2,6)

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

LRTs = np.zeros(len(Ls))
LQs = np.zeros(len(Ls))
Lcohs = np.zeros(len(Ls))
Lcohs_v = np.zeros(len(Ls))
MRT = np.zeros((2,2,len(Ls)))
MQ = np.zeros((2,2,len(Ls)))
Mcoh = np.zeros((2,2,len(Ls)))
Mcoh_v = np.zeros((2,2,len(Ls)))

for theta in Lth:
        print('theta=',theta)
        for alpha in La:
                for zeta in Lz:
                        for iss in range(len(Ls)):
                                s = Ls[iss]   
                                [RTs,LaccS,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,0,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                LRTs[iss] += np.mean(RTs)
                                LQs[iss] += np.mean(Macc)
                                Lcohs[iss] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0))))
                                Lcohs_v[iss] += np.mean(np.abs(np.diff(Macc,axis=1)))
                                for co2 in Lco2:        
                                        for co in Lco:
                                                for kd in Lk:
                                                        [RT,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                        MRT[co2,co-1,iss] += np.mean(RT)
                                                        MQ[co2,co-1,iss] += np.mean(Macc)
                                                        Mcoh[co2,co-1,iss] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0))))
                                                        Mcoh_v[co2,co-1,iss] += np.mean(np.abs(np.diff(Macc,axis=1)))

LRTs = LRTs/nl**3
LQs = LQs/nl**3
Lcohs = Lcohs/nl**3
Lcohs_v = Lcohs_v/nl**3
MRT = MRT/nl**4
MQ = MQ/nl**4
Mcoh = Mcoh/nl**4
Mcoh_v = Mcoh_v/nl**4

np.save('./files/LRTs_s',LRTs)
np.save('./files/LQs_s',LQs)
np.save('./files/Lcohs_s',Lcohs)
np.save('./files/Lcohs_v_s',Lcohs_v)
np.save('./files/MRT_s',MRT)
np.save('./files/MQ_s',MQ)
np.save('./files/Mcoh_s',Mcoh)
np.save('./files/Mcoh_v_s',Mcoh_v)
'''
LRTs = np.load('./files/LRTs_s.npy')
LQs = np.load('./files/LQs_s.npy')
Lcohs = np.load('./files/Lcohs_s.npy')
Lcohs_v = np.load('./files/Lcohs_v_s.npy')
MRT = np.load('./files/MRT_s.npy')
MQ = np.load('./files/MQ_s.npy')
Mcoh = np.load('./files/Mcoh_s.npy')
Mcoh_v = np.load('./files/Mcoh_v_s.npy')'''


names = np.array([['CT','CB'],['PT','PB']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])
lstyles = ['-','--']
lw = 3
# Fig3C
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
ax[0].plot(Ls,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Ls,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\beta$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Ls,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Ls,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\beta$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Ls,Mcoh[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Ls,Lcohs/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\beta$')        
plt.tight_layout()
title = './figures/Metrics_s.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 

################################################################
# SFig2C
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)
ax[0].plot(Ls,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Ls,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\beta$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Ls,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Ls,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\beta$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Ls,Mcoh_v[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Ls,Lcohs_v/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\beta$')        
plt.tight_layout()
title = './figures/Metrics_s_group.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 
plt.show()

