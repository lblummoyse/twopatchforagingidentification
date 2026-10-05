import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from simu_fc import Simu

def get_simu_params():
        B = 0.1 # diffusion coefficient
        eta = 0.5 # counting parameter
        Am = 100 # maximum amount of food in a patch (if depletion)
        tf = 50 # duration of the simulation
        dt = 0.1 # time step
        iDeltar = 1 # >1 if time step between resource distribution times, =1 otherwise 
        idrate = int(1/dt) # sampling period 
        Ttr = 0.2 # travel time
        Lt = np.arange(1e-10,tf,dt) # list of times
        nl = 3 # number of values per parameter
        return [B,eta,Am,tf,dt,iDeltar,idrate,Ttr,Lt,nl]

def get_boundaries():
        thmin = -6
        thmax = -4
        amin = 0.75
        amax = 1.2        
        zmin = 0 
        zmax = 6
        kmin = 0.3 
        kmax = 0.9
        smin = 0
        smax = 0.15
        return [thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax]

def get_P(i,j,l,k,c):
        [B,eta,Am,tf,dt,iDeltar,idrate,Ttr,Lt,nl] = get_simu_params()
        [thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
        kd = kmin + k*(kmax-kmin)/(nl-1)
        theta = thmin + i*(thmax-thmin)/(nl-1)
        alpha = amin + j*(amax-amin)/(nl-1)
        zeta = zmin + l*(zmax-zmin)/(nl-1)        
        s = smin + c*(smax-smin)/(nl-1)
        P = np.array([theta,alpha,zeta,kd,s])
        return P        

def model_samples(co,co2,N,p0,p1,dep,ini,nsimu,theta,alpha,zeta,kd,s):
        [B,eta,Am,tf,dt,iDeltar,idrate,Ttr,Lt,nl] = get_simu_params()
        L = []
        Qin = np.zeros((nsimu,N))
        Qin[:,:ini] = np.ones((nsimu,ini))
        Am = dep 
        [Lavg,Lacc,Macc] = Simu(alpha,theta,B,zeta,p0,p1,iDeltar,N,co,co2,Am,kd,kd,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep)
        L = L + [Macc]
        return np.array(L)

def get_likelihood(co2_,co_,N,nsimu,p0,p1,dep,ini):         # Compute probability distribution
        [thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
        [B,eta,Am,tf,dt,iDeltar,idrate,Ttr,Lt,nl] = get_simu_params()
        Sim = np.zeros((nl,nl,nl,nl,nl,1,nsimu,int(tf/(idrate*dt))))
        nbins = N+1
        bins = np.arange(-0.5*1/N,1+1/N,1/N)
        Prob = np.zeros((nl,nl,nl,nl,nl,1,int(tf/(idrate*dt)),nbins)) # 4 parameters, 2 conditions x 2 group sizes, for tf times, N+1 bins
        for k in range(nl):
                kd = kmin + k*(kmax-kmin)/(nl-1)
                for i in range(nl):
                        theta = thmin + i*(thmax-thmin)/(nl-1)
                        for j in range(nl):
                                alpha = amin + j*(amax-amin)/(nl-1)
                                for l in range(nl):
                                        zeta = zmin + l*(zmax-zmin)/(nl-1)
                                        for c in range(nl):
                                                s = smin + c*(smax-smin)/(nl-1)
                                                P = [theta,alpha,zeta,kd,s]
                                                sim_all = model_samples(co_,co2_,N,p0,p1,dep,ini,nsimu,*P) # nsimu simulations, tf times
                                                for m in range(sim_all.shape[0]):
                                                        for n in range(sim_all.shape[2]): # across tf times
                                                                sim = sim_all[m,:,n] 
                                                                hist_sim,_ = np.histogram(sim, bins=bins, density=True)
                                                                hist_sim = hist_sim/np.sum(hist_sim)
                                                                Prob[i,j,l,k,c,m,n,:] = hist_sim
                                                                
        Prob[Prob==0] = 1e-10
        return Prob

def get_posterior(Prob,obs,N,nl,nsimu_obs):
        [thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
        Prob = np.log(Prob)                                     
        D = np.zeros((nl,nl,nl,nl,nl)) 
        Prod = np.zeros((nl,nl,nl,nl,nl))
        for k in range(nl):
                kd = kmin + k*(kmax-kmin)/(nl-1)
                for i in range(nl):
                        theta = thmin + i*(thmax-thmin)/(nl-1)
                        for j in range(nl):
                                alpha = amin + j*(amax-amin)/(nl-1)
                                for l in range(nl):
                                        zeta = zmin + l*(zmax-zmin)/(nl-1)
                                        for c in range(nl):
                                                s = smin +c*(smax-smin)/(nl-1)
                                                for m in range(obs.shape[0]): # across conditions
                                                        for n in range(obs.shape[2]): # across tf times
                                                                for ns in range(nsimu_obs):
                                                                        ind = int(obs[m,ns,n]*N)                                                        
                                                                        Prod[i,j,l,k,c] = Prod[i,j,l,k,c] + Prob[i,j,l,k,c,m,n,ind]
                                                                             
        return Prod                 

def get_distance(Sim,obs,N,nl,nsimu_obs):
        [thmin,thmax,amin,amax,zmin,zmax,kmin,kmax,smin,smax] = get_boundaries()
        D = np.zeros((nl,nl,nl,nl,nl)) 
        for k in range(nl):
                kd = kmin + k*(kmax-kmin)/(nl-1)
                for i in range(nl):
                        theta = thmin + i*(thmax-thmin)/(nl-1)
                        for j in range(nl):
                                alpha = amin + j*(amax-amin)/(nl-1)
                                for l in range(nl):
                                        zeta = zmin + l*(zmax-zmin)/(nl-1)
                                        for c in range(nl):
                                                s = smin +c*(smax-smin)/(nl-1)
                                                for m in range(obs.shape[0]): # across conditions
                                                        for n in range(obs.shape[2]): # across tf times
                                                                D[i,j,l,k,c] += sc.stats.wasserstein_distance(obs[m,:,n],Sim[i,j,l,k,c,m,:,n])
                                                                
        D = D/obs.shape[2]         
        return D 
                 
def get_MAP(Prod): # Maximum a posteriori       
        arg = np.unravel_index(np.argmax(Prod, axis=None), Prod.shape)        
        P_win = get_P(arg[0],arg[1],arg[2],arg[3],arg[4])
        return arg, P_win
        
def get_error(arg,arg_true):
        arg, _ = get_MAP()
        RE = np.abs(arg - arg_true)
        return RE



