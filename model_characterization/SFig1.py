import sys
sys.path.insert(0, '../functions') 

import matplotlib.pyplot as plt
import numpy as np
from simu_fc import Simu
plt.rcParams.update({'font.size': 24,'text.usetex': True})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import ScalarFormatter
class FixedDecimalFormatter(ScalarFormatter):
    def __init__(self, fformat='%.1f', **kwargs):
        super().__init__(**kwargs)
        self._fformat = fformat

    def _set_format(self):
        self.format = r'$\mathdefault{%s}$' % self._fformat   


B = 0.1 # diffusion coefficient
eta = 1/2 # counting parameter
dep = 0 # 0: no depletion, 1: depletion
Lco2 = [0,1] #0: counting, 1: pulsatile
N = 4 # group size
Lco = [0,1,2] #0: no coupling, 1: threshold modulation, 2: belief modulation
p1 = 0.8 # reward probability in patch 1
ratio = 0.2 # ratio p0/p1
p0 = p1*ratio # reward probability in patch 0

Am = 0 # maximum amount of food (=0 if no depletion)
tf = 50 # duration of the simulation
dt = 0.1 # time step
iDeltar = 1 # >1 if time step between resource distribution times, =1 otherwise 
idrate = int(1/dt) # sampling period 
Ttr = 0.2 # travel time
Lt = np.arange(0,tf,dt) # list of times
nsimu = 4000 # number of simulations

# Recording
MavgS = np.zeros((int(nsimu)))
MaccS = np.zeros((int(len(Lt)/idrate)-1))
MrateS = np.zeros((int(len(Lt)/idrate)))
Mavgso = np.zeros((2,2,int(nsimu)))
Maccso = np.zeros((2,2,int(len(Lt)/idrate)-1))
Mrateso = np.zeros((2,2,int(len(Lt)/idrate)))
Varso = np.zeros((2,2,nsimu,int(len(Lt)/idrate)))

Qin = np.zeros((nsimu,N)) # Initial accuracy
#Qin[:,:int(N/2)] = np.ones((nsimu,int(N/2)))
#Qin  = np.random.randint(2,size=(nsimu,N))#LQin[ip1,iratio,:,:]

# Parameters
[theta,alpha,zeta,kd,s] = [-5 , 0.9 , 3 , 0.5 , 0.05]
for co in Lco:
        if co==0:
                [RT,LaccS,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,0,Am,0,0,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                MaccS = np.mean(np.abs(np.diff(Macc,axis=1)),axis=0)
                VarS = Macc


        elif co>0:
                for co2 in Lco2:
                        print(co2,co)
                        [Lavg,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
                        Maccso[co2,co-1,:] = np.mean(np.abs(np.diff(Macc,axis=1)),axis=0)
                        Varso[co2,co-1,:,:] = Macc

names = np.array([['$\mathrm{CT}$','$\mathrm{CB}$'],['$\mathrm{PT}$','$\mathrm{PB}$']])
colors = np.array([["#0072B2", "#56B4E9"],["#D55E00","#E69F00"]])
plt.figure(figsize = (11,6))
al = 0.05
lstyles = ['-','--']
plt.plot(np.arange(0,tf,idrate*dt)[:-1],MaccS,color='k',linestyle='-',label='$\mathrm{NI}$')
lw = 3
for co2 in Lco2:
        for co in Lco[1:]:
                plt.plot(np.arange(0,tf,idrate*dt)[:-1],Maccso[co2,co-1,:],linewidth=lw,linestyle = lstyles[co-1],color = colors[co2,co-1],label=names[co2,co-1])

al2 = 0.5
lw2 = 3
plt.ylabel(r'$D_{\mathrm{group},t}$')
plt.xlabel(r'$t \, (\mathrm{a.u.})$')     

ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(7,steps=[1, 2, 2.5, 5, 10])) 
plt.legend(loc="upper center", ncols=5,bbox_to_anchor=(0.5, 1.15),frameon=False,handletextpad=0.5,columnspacing=1.5,fontsize = 20)
plt.tight_layout()
plt.savefig('./figures/Fig2_damp.pdf')
        
plt.show()



