import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import numba as nb
import os
from scipy.sparse import csgraph
from scipy.integrate import solve_ivp


# Solve ODE system for disk model with effective hydrodynamic interactions

np.random.seed()

# Video of swimmer dynamics?
mk_vid = 1

# Plot solution;
plot_sol = 1

# Path to save all data
sv_file = 'Data/'

# Simulation ID

simID = "sym30_2000_500_100_OmegaDelay_UNS"
print("sim ID: "+simID)

# Simulation time for ODE model
simtimes = np.linspace(0, 2000, 1000)

N = 500

# Size of periodic box (in units of embryo size)
L = 100

# Periodic domain?
per_dom = 0

# Compressing surface potential?
Surf_Pot = 0
WellLength = 30  # Length scale of the well potential

# Stokeslet strength (must be >0, sets the attraction strength between disks)
Fg = 219 + 70 *  np.random.randn(N)  # Units: [Embryo-Radius^2]/second

# Maximum interaction distance for attractive Stokeslet interaction
RFg_int = 3.8  # 2*sqrt(2) is the second next nearest neighbour in hexagonal grid

# Strength of rotational near-field interactions of neighbouring particles
# Free spinning calibration
f0 = -0.06
tau0 = 0.12

# Minimal distance of disk boundaries from which near-field interactions start
Rnf_int = 0.5

# Single disk angular frequency (= time-scale)
omega0 = 0.05 * 2 * np.pi * (0.72 + 0.17 * np.random.randn(N))
omega_last = omega0.copy()
OmegaDelay = 1
    

# Flow interactions between cells
# = 0: each cell will only interact with its image
# = 1: each cell interacts with all other cells and with its image
Flowinteract = 1

# Lateral steric repulsion
Sterinteract = 1

# Spin-Frequency near-field interactions to slow down nearby spinners?
NFinteract = 1

# Symmetrize the flow interactions?
# Unsymmetrization cause noise which can drive the translational motion
Symmetrize = 0

# Far-field attraction from embryos with up to two neighbours
# (otherwirse only near-field interactions)
SelectFarField = 1

# Modulation of intrinsic torques through presence of nearby embryos
ModOmega0 = 1
N0_damping = 80

###### GRAVITY (=Stokeslet direction) (DON'T CHANGE IN DISK MODEL) ######
grav_vec = np.array([0, 0, -1])
grav_vec = grav_vec / np.linalg.norm(grav_vec)

# Distance of flow singularity below the surface
h = 1

# Strength of steric repulsion
Frep_str = 3200 * (1 + 0.4 * (np.random.rand(N) - 0.5)) # For 1/r^12 repulsive potential (CORRECT ONE)

###################### SET INITIAL POSITIONS ############################

# % %%%%%%% RANDOM INITIAL CONDITIONS (PAPER) %%%%%%%
# % %%%%% (Starts with particles far outside) %%%%%%%
phi_part = 2 * np.pi * np.random.rand(N)  # Random angles
R_part = 0.8 * (350 + 200 * np.sqrt(np.random.rand(N)) ** (1 / 6))  # Random radii
Pos_init = np.column_stack((R_part * np.cos(phi_part), R_part * np.sin(phi_part)))
# % % Apply random stretch factors
for i in range(1):
    rand_stretch = 1 + 0.4 * np.random.randn(N)
    Pos_init = np.column_stack((rand_stretch, rand_stretch)) * Pos_init
# % % Move to center of domain
Pos_init = Pos_init - np.mean(Pos_init, axis=0)
Pos_init = Pos_init % L
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# % % Plot initial positions
fig, ax = plt.subplots()
xdata = Pos_init[:,0]
ydata = Pos_init[:,1]
ax.clear()
ax.scatter(xdata, ydata, c='r', s=3, edgecolors='black')
plt.show()

# % % Flatten for ode solver
Pos_init = Pos_init.reshape(-1,1)

################### END SET INITIAL POSITIONS ###########################

###################### Singularity implementation  ######################
# Functions require input of position vectors [rx1, ry1, rz1; rx2, ry2, rz2; ...]
# and similarly for the parameterizing Stokeslet unit vector f
@nb.njit
def norm2d(arr, ax):
    return np.sqrt((arr ** 2).sum(axis=ax))

################%% STOKESLET flow %%%%%%%%%%%%%%%%%%
@nb.njit
def u_st(f, r):
    r_norm = norm2d(r, 1).reshape(-1, 1)
    return (1 / (8 * np.pi)) * (f / r_norm +
                                r * np.sum(f * r, axis=1).reshape(-1, 1) / (r_norm ** 3).reshape(-1, 1))

################% Steric repulsion of nearby embryos %%%%%%%%%%%%%%%%%%%%%
# 1/r^12 potential repulsion between midpoints -> 1/r^13 force
# (written \vec{r}/r^14 because of vec(r) in nominator!!)
@nb.njit
def Frep(r):
    r_norm = norm2d(r, 1).reshape(-1, 1)
    return 12 * r / r_norm ** 14

############### Radial dependence of near-field forces %%%%%%%%%%%%%%%%%##
@nb.njit
def feta(r):
    return np.log(Rnf_int / np.abs(r - 2))

############## Radial dependence of near-field torques %%%%%%%%%%%%%%%%%##
@nb.njit
def taueta(r):
    return np.log(Rnf_int / np.abs(r - 2))

# Transformations of image vector and pseudo-vector orientations
@nb.njit
def vec_img(e):
    return np.column_stack((e[:, 0], e[:, 1], -e[:, 2]))  # Stokeslet, force- and source-dipole

def EmbryoDynamics(t, y_):
    global omega_last
    # Print time for progress
    if t % 0.001 < 0.00005:
        print(t)
    if per_dom == 1:
        # Periodic boundary conditions
        y = y_ % L
    else:
        y = y_
    # ODE function of embryo dynamics for axisymmetric embryos:
    # Input vector y contains for each particle 2D position and 3D orientation
    # [x1; y1; x2; y2; ... xN; yN]
    # Extract and format positions as needed for flow functions
    Pos3D = y[:2 * N].reshape((N, 2))
    Pos3D = np.column_stack((Pos3D, -h * np.ones(N)))

    if Surf_Pot == 1:
        # Cylindircal coordinates of the particle positions
        Pos_R = Pos3D[:, :2] - [L / 2, L / 2]
        R_pos = np.linalg.norm(Pos_R, axis=1)
        Phi_pos = np.pi + np.arctan2(-Pos_R[:, 1], -Pos_R[:, 0])
    
    # Signed distance matrices r_i - r^0_i where flows from
    # singularities placed at r^0_i are evaluated at r_i
    dist_x = Pos3D[:, 0].reshape(1, -1) - Pos3D[:, 0].reshape(-1, 1)
    dist_y = Pos3D[:, 1].reshape(1, -1) - Pos3D[:, 1].reshape(-1, 1)
    # Stokeslet force orientation (DON'T CHANGE FOR DISK MODEL)
    grav = np.tile(grav_vec, (N, 1))
    grav = grav / np.linalg.norm(grav, axis=1).reshape(-1, 1)

    # Fixed global orientation of gravity
    fst = grav
    fst_img = vec_img(fst)

    # Determine angular frequency of each particle
    rij = np.sqrt(dist_x ** 2 + dist_y ** 2)  # Distance matrix
    NH_matrix = (rij < (2 + Rnf_int)).astype(int)  # Adjacency matrix of particles within near-field interaction distance
    r,p = csgraph.connected_components(NH_matrix, directed=False)
    NH_matrix = NH_matrix - np.eye(N)
    omega_all = omega0.copy()

    if SelectFarField == 1:
        # Assume all particles participate in far-field interactions
        idx_FF = np.ones(N, dtype=bool)

    # ANGULAR FREQUENCY CALCULATION
    if NFinteract == 1:
        if OmegaDelay == 1 and t != 0:
            sum_count = np.zeros(N)
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
                    if (OmegaDelay == 1 and t != 0):
                        sum_count[curr_disk] = 1 + np.sum(tau)
                    else:
                        M[l,l] = 1 + np.sum(tau)
                    for n in range(len(nh_vec)):
                        M[l, np.where(idx_num==nh_vec[n])[0]] = tau[n]
                    
                if ModOmega0 == 1:
                    # Renormalize intrinsic rotation frequencies
                    omega0_M = omega0[idx_num] / (1 + (nrd_cc / N0_damping)**2)
                # Solve for the angular frequencies in this connected component
                omega = (omega0_M - M @ omega_last[idx_num]) / sum_count[idx_num] if (OmegaDelay == 1 and t != 0) else np.linalg.solve(M, omega0_M)
                # Add into array of all angular frequencies according to disk IDs
                omega_all[idx_num] = omega

                if SelectFarField == 1:
                    # If these particle belong to group with more than 3 particles
                    # remove those indices from the far-field interaction list
                    if len(idx_num) > 3:
                        idx_FF[idx_num] = False
    
    # If OmegaDelay is included
    if OmegaDelay == 1:
        omega_last = omega_all

    if Surf_Pot == 1:
        # Emulate centering effect of well curvature
        upot_x = -WellLength ** (-2) * (R_pos) * np.cos(Phi_pos)
        upot_y = -WellLength ** (-2) * (R_pos) * np.sin(Phi_pos)

    # Sum up all flow contributions that affect a given disk
    # Cross-product vectors for near-field force interactions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RotForce_dir_x = dist_y / rij
        RotForce_dir_y = -dist_x / rij

    u_e = np.zeros((N, 2))  # Initiate translational velocities array

    for j in range(N):  # Compute velocity for each particle
        if Flowinteract == 1:  # Full flow interactions
            # Logical index for particles whose lateral interaction is taken into account
            idx_lat = np.ones(N, dtype=bool)
            idx_lat[j] = False

            if SelectFarField == 1 and idx_FF[j]:
                # If particle is in far-field group (at most one neighbour)
                # it will interact with all particles
                pass
            else:
                # Find all particles further than a distance away
                # and exclude them from Stokeslet interactions
                idx_far = rij[:, j] > RFg_int
                idx_lat[idx_far] = False

            # Logical index for particles whose image interaction is taken into account
            idx_img = idx_lat  # Only images that participate in lateral interactions
        else:  # Each cell only interacts with its image
            idx_lat = np.zeros(N, dtype=bool)  # No lateral interactions
            idx_img = np.zeros(N, dtype=bool)
            idx_img[j] = True  # Cell interacts only with its image

        if Sterinteract == 1:
            idx_ster = idx_lat  # Same neighbourhood as Stokeslet interaction

        # Because all particles are at the same distance below
        # the surface only image flow interaction play a role
        r_curr = np.column_stack((dist_x[idx_img, j], dist_y[idx_img, j], -2 * h * np.ones(np.sum(idx_img))))

        # Collect distances for steric interactions
        if Sterinteract == 1:
            # Same neighbourhood as Stokeslet interaction
            r_curr_ster = r_curr.copy()
            r_curr_ster[:, 2] = 0

        # Prepare according arrays of vectors parameterizing flow singularities
        fst_curr = fst_img[idx_img]
        # Collect all attractive Stokeslet flow interactions (no additional weighting)
        if Symmetrize == 1:
            u_star = 0.5 * np.sum(np.tile((Fg[j] + Fg[idx_img]).reshape(-1,1),(1,3)) * u_st(fst_curr, r_curr),axis=0)  # SYMMETRIZED
        else:
            u_star = np.sum(np.tile((Fg[idx_img]).reshape(-1,1),(1,3)) * u_st(fst_curr, r_curr),axis=0)  # UNSYMMETRIZED
        u_e[j] = u_star[:2]  # Only vx and vy are relevant
        
        # Steric repulsion only laterally between embryos
        if Sterinteract == 1:
            u_rep = 0.5 * np.sum((Frep_str[j] + Frep_str[idx_ster]).reshape(-1,1) * Frep(r_curr_ster), axis=0)
            u_e[j] += u_rep[:2]
        
        # Contributions from transverse force interactions
        idx_neighb = NH_matrix[j, :] > 0  # Neighbour-indices (no diagonals!)
        if np.sum(idx_neighb) != 0:
            nr_neighb = np.sum(idx_neighb)  # Number of neighbours
            feta_curr = feta(rij[j, idx_neighb])

            # Build omega array that can be used for rotation force summation
            omega_full = np.tile(omega_all[j], (nr_neighb, 1)).T + omega_all[idx_neighb]

            # Sum up all transverse force contributions for given particles
            u_e[j, 0] += f0 * np.sum(feta_curr * omega_full * RotForce_dir_x[j, idx_neighb])
            u_e[j, 1] += f0 * np.sum(feta_curr * omega_full * RotForce_dir_y[j, idx_neighb])
    # Fill RHS output vector of ODE system
    dydt = u_e.flatten().T

    # If well curvature effect is included
    if Surf_Pot == 1:
        u_pot = np.column_stack((upot_x, upot_y)).T
        dydt += u_pot.flatten()

    return dydt


# Solve ODEs
start_time = time.time()
ysol = solve_ivp(EmbryoDynamics, [simtimes[0], simtimes[-1]], Pos_init.flatten(), method="RK23", t_eval=simtimes, rtol=1e-3, atol=1e-3)
end_time = time.time()
execution_time = end_time - start_time
print("Total time: ", execution_time)
print(ysol)

# Save data
directory = os.path.dirname(sv_file+simID+'/')
if not os.path.exists(directory):
    os.makedirs(directory)
if per_dom == 1:
    np.save(sv_file + simID +'/' + simID + '_pos.npy', ysol.y % L)
else:
    np.save(sv_file + simID +'/' + simID + '_pos.npy', ysol.y)
np.save(sv_file + simID +'/' + simID + '_time.npy', ysol.t)
np.save(sv_file + simID +'/' + simID + '_omega0.npy', omega0)
np.save(sv_file + simID +'/' + simID + '_params.npy', [N, L, Rnf_int, tau0, ModOmega0, N0_damping, NFinteract])
