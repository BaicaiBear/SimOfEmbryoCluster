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

@nb.njit
def norm2d(arr, ax):
    return np.sqrt((arr ** 2).sum(axis=ax))

@nb.njit
def u_st(f, r):
    r_norm = norm2d(r, 1).reshape(-1, 1)
    return (1 / (8 * np.pi)) * (f / r_norm +
                                r * np.sum(f * r, axis=1).reshape(-1, 1) / (r_norm ** 3).reshape(-1, 1))

@nb.njit
def Frep(r):
    r_norm = norm2d(r, 1).reshape(-1, 1)
    return 12 * r / r_norm ** 14

@nb.njit
def feta(r):
    return np.log(Rnf_int / np.abs(r - 2))

@nb.njit
def taueta(r):
    return np.log(Rnf_int / np.abs(r - 2))

@nb.njit
def vec_img(e):
    return np.column_stack((e[:, 0], e[:, 1], -e[:, 2]))

def ComputeStress(t):
    dist_x = Pos[:,t].reshape(-1, 2)[:, 0].reshape(1, -1) - Pos[:,t].reshape(-1, 2)[:, 0].reshape(-1, 1)
    dist_y = Pos[:,t].reshape(-1, 2)[:, 1].reshape(1, -1) - Pos[:,t].reshape(-1, 2)[:, 1].reshape(-1, 1)
    omega = omega_alltime[:,t]
    rij = np.sqrt(dist_x ** 2 + dist_y ** 2)  # Distance matrix
    NH_matrix = (rij < (2 + Rnf_int)).astype(int) - np.eye(N) # Adjacency matrix of particles within near-field interaction distance
    # INTERNAL STRESS CALCULATION
    outp = np.zeros((N, 3))
    for j in range(N): # Loop through connected components      
        nh_vec = np.where(NH_matrix[j, :] > 0)[0]
        

