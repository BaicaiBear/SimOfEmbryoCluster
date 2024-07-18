import numpy as np
import scipy.sparse.csgraph as csgraph
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.colors import LogNorm

simID = 'sym'

MK_video = 1

### Load ode solution ###
Pos = np.load('Data/'+simID+'/'+simID+'_pos.npy')
time = np.load('Data/'+simID+'/'+simID+'_time.npy')
omega_alltime = np.load('Data/'+simID+'/'+simID+'_omega_alltime.npy')
[N, L, Rnf_int, tau0, ModOmega0, N0_damping, NFinteract] = np.load('Data/'+simID+'/'+simID+'_params.npy')
N = int(N)

