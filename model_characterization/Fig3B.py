import sys
sys.path.insert(0, '../functions') 
sys.path.insert(0, '../model_selection_identifiability')  

import matplotlib.pyplot as plt
import numpy as np
from simu_fc import Simu
from Bay_fc import get_boundaries
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import ScalarFormatter
plt.rcParams.update({'font.size': 24,"text.usetex": True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

class FixedDecimalFormatter(ScalarFormatter):
    def __init__(self, fformat="%.1f", **kwargs):
        super().__init__(**kwargs)
        self._fformat = fformat

    def _set_format(self):
        self.format = r"$\mathdefault{%s}$" % self._fformat   
        
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
Ls = np.linspace(smin,smax,nl)
Lk = np.linspace(kmin,kmax,nl)
Lz = np.linspace(0,8,5)

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

LRTs = np.zeros(len(Lz))
LQs = np.zeros(len(Lz))
Lcohs = np.zeros(len(Lz))
Lcohs_v = np.zeros(len(Lz))

MRT = np.zeros((2,2,len(Lz)))
MQ = np.zeros((2,2,len(Lz)))
Mcoh = np.zeros((2,2,len(Lz)))
Mcoh_v = np.zeros((2,2,len(Lz)))


for theta in Lth:
        print('theta=',theta)
        for alpha in La:
                        for s in Ls:
                                for iz in range(len(Lz)):
                                        zeta = Lz[iz]   
                                        [RTs,LaccS,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,0,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                        LRTs[iz] += np.mean(RTs)
                                        LQs[iz] += np.mean(Macc)
                                        Lcohs[iz] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0))))
                                        Lcohs_v[iz] += np.mean(np.abs(np.diff(Macc,axis=1)))
                                        for kd in Lk:
                                                for co2 in Lco2:        
                                                        for co in Lco:
                                                                
                                                        
                                                                [RT,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                                                                if len(RT)==0:
                                                                        RT = np.array([tf])
                                                                MRT[co2,co-1,iz] += np.mean(RT)
                                                                MQ[co2,co-1,iz] += np.mean(Macc)
                                                                Mcoh[co2,co-1,iz] += np.mean(np.abs(np.diff(np.mean(Macc,axis=0))))
                                                                Mcoh_v[co2,co-1,iz] += np.mean(np.abs(np.diff(Macc,axis=1)))
co = ['tab:purple','tab:blue','tab:green']

print(MRT)

LRTs = LRTs/nl**3
LQs = LQs/nl**3
Lcohs = Lcohs/nl**3
Lcohs_v = Lcohs_v/nl**3
MRT = MRT/nl**4
MQ = MQ/nl**4
Mcoh = Mcoh/nl**4
Mcoh_v = Mcoh_v/nl**4

np.save('./files/LRTs_z',LRTs)
np.save('./files/LQs_z',LQs)
np.save('./files/Lcohs_z',Lcohs)
np.save('./files/Lcohs_v_z',Lcohs_v)
np.save('./files/MRT_z',MRT)
np.save('./files/MQ_z',MQ)
np.save('./files/Mcoh_z',Mcoh)
np.save('./files/Mcoh_v_z',Mcoh_v)
'''
LRTs = np.load('./files/LRTs_z.npy')
LQs = np.load('./files/LQs_z.npy')
Lcohs = np.load('./files/Lcohs_z.npy')
Lcohs_v = np.load('./files/Lcohs_v_z.npy')
MRT = np.load('./files/MRT_z.npy')
MQ = np.load('./files/MQ_z.npy')
Mcoh = np.load('./files/Mcoh_z.npy')
Mcoh_v = np.load('./files/Mcoh_v_z.npy')'''

names = np.array([['CT','CB'],['PT','PB']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])
lstyles = ['-','--']
lw = 3
#Fig3B
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[2].yaxis.set_major_formatter(FixedDecimalFormatter("%.1f", useMathText=True))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
ax[0].plot(Lz,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Lz,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\overline{\zeta}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Lz,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Lz,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\overline{\zeta}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Lz,Mcoh[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Lz,Lcohs/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\overline{\zeta}$')        
plt.tight_layout()
title = './figures/Metrics_z.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 

#####################################################
# SFig2B
fig, ax = plt.subplots(1,3,figsize=(18,3.5),sharex=True,sharey=False)
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
ax[2].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1), useMathText=True)
ax[0].plot(Lz,LRTs,color='black',label='NI',marker='o',linewidth=lw,linestyle = '-') 
for co2 in Lco2:        
        for co in Lco:
                ax[0].plot(Lz,MRT[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[0].set_ylabel(r'$T_\mathrm{res} \, (\mathrm{a.u.})$')
ax[0].set_xlabel(r'$\overline{\zeta}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[1].plot(Lz,MQ[co2,co-1,:],color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[1].plot(Lz,LQs,color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[1].set_ylabel(r'$Q$')
ax[1].set_xlabel(r'$\overline{\zeta}$')        
for co2 in Lco2:        
        for co in Lco:
                ax[2].plot(Lz,Mcoh_v[co2,co-1,:]/(idrate*dt),color=colors[co2,co-1],label=names[co2,co-1],marker='o',linewidth=lw,linestyle = lstyles[co-1])
ax[2].plot(Lz,Lcohs_v/(idrate*dt),color='black',label='NI',marker='o',linewidth=lw,linestyle = lstyles[0]) 
ax[2].set_ylabel(r'$D_\mathrm{group} \, (\mathrm{a.u.}^{-1})$')
ax[2].set_xlabel(r'$\overline{\zeta}$')        
plt.tight_layout()
title = './figures/Metrics_z_group.pdf'
fig.subplots_adjust(wspace=0.4,hspace=0.)
plt.savefig(title,bbox_inches='tight') 
              
plt.show()

