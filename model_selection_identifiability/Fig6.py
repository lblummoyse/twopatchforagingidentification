import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 21,'text.usetex': False,'font.family': 'serif','font.serif': ['cmr10'],'mathtext.fontset': 'cm',
'axes.formatter.use_mathtext': True,'axes.unicode_minus': False,})

N = 4; nsimu_obs=20; ratio=0.2; dep=0; ini=0 # Experimental conditions

method = 'dis'#'Bay' # estimation method, Bay: Bayesian inference, dis: Wasserstein distance minimization

## name for results based on distribution of departure times:
#fname = './files/conf_matrix_P_'+method+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.npy'

## name for results based on distributions of accuracy:
fname = './files/conf_matrix_'+method+'_'+str(N)+'_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.npy'

Win = np.load(fname)
Win = Win/np.sum(Win[0,:])
fig,ax = plt.subplots(1,1)
imax = ax.imshow(Win,vmin=0,vmax=1)
models_name = ['NI','CT', 'CB','PT', 'PB']
ax.set_xticks(range(5), labels=models_name)
ax.xaxis.tick_top()
ax.tick_params(length=0)  
ax.set_yticks(range(5), labels=models_name)
for i in range(5):
    for j in range(5):
        if i==j:
                color = 'k'
        else:
                color = 'w'
        text = ax.text(j,i,np.round(Win[i, j],2),ha='center',va='center',color=color)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')  
plt.tight_layout()

## name for results based on distribution of departure times:
#title = './figures/Conf_P_'+method+'_'+str(N)+'_nsimu_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.pdf'

## name for results based on distributions of accuracy:
title = './figures/Conf_'+method+'_'+str(N)+'_nsimu_'+str(nsimu_obs)+'_env_'+str(int(ratio*10))+'_dep_'+str(dep)+'_'+str(ini)+'.pdf'

fig.savefig(title)
plt.show()                      





