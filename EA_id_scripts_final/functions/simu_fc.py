import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  
def Simu(alpha_,theta0_,B_ ,zeta_,p0,p1,iDeltar,N_,co,co2,Am,kd0_,kd1_,s,eta,idrate,Ttr,Lt,dt,nsimu,Qin,dep):
        N = int(N_*nsimu) #''Parallel'' computing
        # distribution of parameters and boundaries (if large deviation)
        alpha = np.random.normal(alpha_,(alpha_)*s,N)
        alpha = alpha*np.heaviside(alpha,0)
        theta0 = np.random.normal(theta0_,(-theta0_)*s,N)
        theta0 = theta0*np.heaviside(-theta0,0)
        B = np.random.normal(B_,(B_)*s,N)
        B = B*np.heaviside(B,0)
        zeta = np.random.normal(zeta_,(zeta_)*s,N)
        zeta = zeta*np.heaviside(zeta,0)        
        kd0 = np.random.normal(kd0_,(kd0_)*s,N)
        kd0 = kd0*np.heaviside(kd0,0)
        kd1 = kd0
        if co==0: # non-interacting
                kd0_ = 0
                kd0 = 0
                kd1 = 0
        # Connection matrix
        M = np.zeros((N,N))
        for ns in range(nsimu):
                M[ns*N_:(ns+1)*N_,ns*N_:(ns+1)*N_] = np.ones((N_,N_))       
        
        sqrt_bis = np.sqrt(2*B*dt)
        # Recording
        Mpatch1 = np.zeros((nsimu,int(len(Lt)/idrate)))
        Lavg1 = np.zeros(nsimu)
        C = np.zeros(N)
        if dep>0: # if depletion
                A0 = p0*Am
                A1 = p1*Am
                p0_ = p0
                p1_ = p1
        theta = theta0*np.ones(N) # initial threshold
        X = np.zeros(N) # initial decision variable
        Qin = np.squeeze(Qin.reshape((N, 1))) # initial accuracy
        patch = Qin
        D_0to1 = np.zeros(N); D_1to0 = np.zeros(N)  # departure record      
        p = p1*np.heaviside(patch-0.5,0) + p0*np.heaviside(0.5-patch,0) # initial reward probabilities
        LTlast = np.zeros(N) # record last departure time for each agent
        ref = np.ones(N)
        LTlastrw = np.zeros(N)
        refrw = np.ones(N)
        Trw = 1
        aom = 1.5
        Lavg1p = np.zeros(N)
        #LX = np.zeros(len(Lt)); Lrw = []; T0 = [] # if recording
        y = 0.5*np.ones(N)
        Cx = 0; Cth = 0 # social couplings
        # learning variables
        y0 = np.zeros(N); y1 = np.zeros(N)
        n0 = np.zeros(N); n1 = np.zeros(N)
        y0m = np.zeros(N); y1m = np.zeros(N)
        nm0 = np.zeros(N); nm1 = np.zeros(N)
        rw = np.zeros(N)
        om_b = np.zeros(N)
        
        a = 0.001 # replenishment factor
        LT = np.array([])
        
        for it in range(len(Lt)):
                t = Lt[it] # time
                #LX[it] = X[0] # if recording
                X_old = X.copy()   
                X = X + (dt*(rw - alpha) + sqrt_bis*np.random.randn(N))*ref          
                # food rewards
                rw = np.zeros(N)
                if it%iDeltar==0 and it>0:
                        pr = np.random.rand(N)/dt                        
                        rw = np.heaviside(p-pr,1)*refrw*Trw/dt
                        #if rw[0]>0: # if recording
                         #       Lrw.append(t) 
                        LTlastrw = LTlastrw*(1-np.heaviside(rw,0)) + t*np.heaviside(rw,0)
                        refrw = np.heaviside(t-LTlastrw-Trw,1)                        
                        y0 += rw*(1-patch)*ref
                        n0 += np.ones(N)*ref
                        y1 += rw*patch*ref
                        n1 += np.ones(N)*ref
                        if dep>0: # if depletion
                                Sr0 = dt*(rw*(1-patch)).reshape(-1, N_).sum(axis=1).repeat(N_)
                                Sr1 = dt*(rw*patch).reshape(-1, N_).sum(axis=1).repeat(N_)
                                A0 = A0-Sr0*np.heaviside(A0,0) + a*np.heaviside(p0*Am-A0,0)
                                A1 = A1-Sr1*np.heaviside(A1,0) + a*np.heaviside(p1*Am-A1,0)
                                A0 = A0*np.heaviside(A0,0)
                                A1 = A1*np.heaviside(A1,0)
                                p0_ = A0/Am
                                p1_ = A1/Am
                                p = p1_*np.heaviside(patch-0.5,0) + p0_*np.heaviside(0.5-patch,0) 
      
                change = np.heaviside(theta-X,0) # if a threshold is reached, the agent change for the next patch
                if it%idrate==0:
                        if int(it/idrate)<len(Mpatch1[0,:]):
                                if it==0:       
                                        Mpatch1[:,int(it/idrate)] = patch.reshape(-1, N_).mean(axis=1)
                                else:
                                        Mpatch1[:,int(it/idrate)] = patch.reshape(-1, N_).mean(axis=1)
                Lavg1p += patch
                #if departure
                D_0to1 = np.zeros(N); D_1to0 = np.zeros(N)
                if (change>0).any():                            
                        if (change*patch>0).any():
                                LT = np.concatenate([LT,t-LTlast[np.nonzero(patch*change)]]) # list of residence times for model simu first part
                                #LT = np.concatenate([LT,[t]*int(np.sum(patch*change))]) #list of departure times for P metrics in identification
                        LTlast = LTlast*(1-change) + t*change # update the vector of last departure times
                        #if change[0]==1: # if recording
                         #       T0.append(it)
                        if co2==1: # Pulsatile
                                D_0to1 = np.matmul(M,change*(1-patch))/(N_-1)
                                D_1to0 = np.matmul(M,change*patch)/(N_-1)
                                # alternative normalization:
                                #nd0 = np.matmul(M,1-patch) # number of agents in a patch
                                #nd1 = np.matmul(M,patch)
                if kd0_>0:
                        if co2==0: # counting
                        
                                C = (kd0*((np.matmul(M,1-patch)-1)/(N_-1)-eta)*(1-patch) + kd1*((np.matmul(M,patch)-1)/(N_-1)-eta)*patch)
                                C = np.nan_to_num(C)
                        elif co2==1: # pulsatile
                                D0d = D_0to1*(1-patch)
                                D1a = D_0to1*patch
                                D1d = D_1to0*patch
                                D0a = D_1to0*(1-patch)
                                
                                C =  (- kd0*(D0d+D1d) + kd1*(D0a+D1a))
                        if co==1: # threshold modulation
                                Cx = 0
                                Cth = C 
                        elif co==2: # belief modulation
                                Cx = C 
                                if co2==0:
                                        Cx = Cx*dt
                                Cth = 0	
                        elif co==0:
                                Cx = 0
                                Cth = 0
                        if co2==1: # pulsatile kappa scaling
                                Cx = Cx*4
                                Cth = Cth*3
               
                X = X + Cx      
                theta = theta0*np.ones(N) - (-theta0)*Cth*ref 
                # boundaries
                theta = theta*np.heaviside(-theta-1.75,0) - 1.75*(1-np.heaviside(-theta-1.75,0))
                theta = theta*np.heaviside(theta+8.25,0) - 8.25*(1-np.heaviside(theta+8.25,0))
                # learning
                y0m += np.nan_to_num(y0/n0*change*(1-patch))
                y1m += np.nan_to_num(y1/n1*change*patch)
                nm0 += change*(1-patch)
                nm1 += change*patch
                y0 = y0*(1-change)
                n0 = n0*(1-change)
                y1 = y1*(1-change)
                n1 = n1*(1-change)
                H = np.nan_to_num(y1m/nm1)-np.nan_to_num(y0m/nm0)
                om_b_ = -H*patch + H*(1-patch)
                om_b_ = zeta*om_b_
                # learning boundaries
                om_b_ = om_b_*np.heaviside(om_b_+aom,0) - aom*(1-np.heaviside(om_b_+aom,0))
                om_b_ = om_b_*np.heaviside(-om_b_+aom,0) + aom*(1-np.heaviside(-om_b_+aom,0))
                om_b = om_b*(1-change) + om_b_*change  	  
                X = X*(1-change)*ref + om_b*((1-ref)+change)                          
                patch = patch + (-1*np.floor(patch==1) + np.floor(patch==0))*change  
                if dep>0:
                        p = p1_*np.heaviside(patch-0.5,0) + p0_*np.heaviside(0.5-patch,0)
                else:
                        p = p1*np.heaviside(patch-0.5,0) + p0*np.heaviside(0.5-patch,0)     
                # traveling time
                ref = np.heaviside(t-LTlast-Ttr,1) # update refractory variable, = 0 during traveling periods, 1 otherwise
        Lavg1 = Lavg1p.reshape(-1, N_).mean(axis=1)/len(Lt)	
        Lpatch1 = np.mean(Mpatch1,axis=0)
        return [LT,Lpatch1,Mpatch1]
            




