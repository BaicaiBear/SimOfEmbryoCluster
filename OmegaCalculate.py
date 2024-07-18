import numpy as np
import scipy.sparse.csgraph as csgraph
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.colors import LogNorm

simID = 'sym20_1000_200_100_S'

MK_video = 1

### Load ode solution ###
Pos = np.load('Data/'+simID+'/'+simID+'_pos.npy')
time = np.load('Data/'+simID+'/'+simID+'_time.npy')
omega0 = np.load('Data/'+simID+'/'+simID+'_omega0.npy')
[N, L, Rnf_int, tau0, ModOmega0, N0_damping, NFinteract] = np.load('Data/'+simID+'/'+simID+'_params.npy')
N = int(N)

@nb.njit
def taueta(r):
    return np.log(Rnf_int / np.abs(r - 2))

def ComputeOmega(x,y,omega0):
    dist_x = x.reshape(1, -1) - x.reshape(-1, 1)
    dist_y = y.reshape(1, -1) - y.reshape(-1, 1)
    rij = np.sqrt(dist_x ** 2 + dist_y ** 2)  # Distance matrix
    NH_matrix = (rij < (2 + Rnf_int)).astype(int)  # Adjacency matrix of particles within near-field interaction distance
    r,p = csgraph.connected_components(NH_matrix, directed=False)
    NH_matrix = NH_matrix - np.eye(N)
    outp = omega0.copy().reshape(-1,1)
    # ANGULAR FREQUENCY CALCULATION
    if NFinteract == 1:
        for j in range(r): # Loop through connected components      
            idx_num = []
            if np.count_nonzero(p==j) > 1: # If at least two elements in connected component
                # Extract numeric indices of all disks in this component
                idx_num = np.where(p == j)[0]
                # print(idx_num)
            if len(idx_num):
                # Number of disks in current connected component
                nrd_cc = len(idx_num)
                
                # # Sorted index vector needed to fill linear system matrix
                # idx_lin = list(range(nrd_cc))
                
                # Linear matrix of the torque balance for given connected component
                M = np.zeros((nrd_cc, nrd_cc))
                
                # Loop through those disks to build linear system
                for l in range(nrd_cc):
                    # Current disk
                    curr_disk = idx_num[l]
                    
                    # Near-field interaction neighbours for current disk
                    nh_vec = np.where(NH_matrix[curr_disk, :] > 0)[0]
                    
                    # Pair-wise distances of disks within interaction distance
                    rij_curr = rij[curr_disk, nh_vec]
                    
                    # Torque interactions strengths for those distances
                    tau = tau0 * taueta(rij_curr)
                    
                    # Fill linear system matrix row for given particle
                    M[l, l] = 1 + np.sum(tau)
                    for n in range(len(nh_vec)):
                        M[l, np.where(idx_num==nh_vec[n])[0]] = tau[n]

                if ModOmega0 == 1:
                    # Renormalize intrinsic rotation frequencies
                    omega0_M = omega0[idx_num] / (1 + (nrd_cc / N0_damping)**2)
                else:
                    omega0_M = omega0[idx_num]
                # Solve for the angular frequencies in this connected component
                omega = np.linalg.solve(M, omega0_M)
                # Add into array of all angular frequencies according to disk IDs
                outp[idx_num] = omega.reshape(-1,1)
    return outp


omega_alltime = np.array(omega0).reshape(-1,1) # Angular frequencies of all disks at all times
# Compute angular frequencies at all times
for i in range(len(time)):
    xdata = Pos[:,i].reshape(-1, 2)[:, 0] 
    ydata = Pos[:,i].reshape(-1, 2)[:, 1] 
    omega = ComputeOmega(xdata,ydata,omega0)
    omega_alltime = np.column_stack((omega_alltime,omega))
omega_alltime = np.delete(omega_alltime, 0, 1)
np.save('Data/'+simID+'/'+simID+'_omega_alltime.npy', omega_alltime)

### Plotting ###
omega_alltime = np.abs(omega_alltime) # Absolute value for log scale
fig, ax = plt.subplots()
sc = ax.scatter([], [], c=[], cmap='jet', s=10)
cbar = plt.colorbar(sc)
cbar.set_label('Omega')
cbar.mappable.set_norm(LogNorm(vmin=1e-5, vmax=1))
def update(frame):
    xdata = Pos[:,frame].reshape(-1, 2)[:, 0] 
    ydata = Pos[:,frame].reshape(-1, 2)[:, 1] 
    cdata = omega_alltime[:, frame]
    sc.set_offsets(np.c_[xdata, ydata])
    sc.set_array(cdata)
    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 120)
    return sc, 
ani = animation.FuncAnimation(fig, update, frames=range(len(time)), interval=20)

# Save animation as video
if MK_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('Data/'+simID+'/'+simID+'_animation.mp4', writer=writer)

plt.show()
