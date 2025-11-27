'''
Docstring for data.lbm

This module implements a simple D2Q9 Lattice-Boltzmann scheme with a naive driving force.
'''
import numpy as np
import os
path = os.path.dirname(__file__)
from numba import njit

@njit(fastmath=True)
def big_LBM(solid, T, force_dir):
    """
    A Lattice Boltzmann simulation.
    This single function inlines equilibrium, collision, streaming,
    bounce-back, and macroscopic updates to optimize njit performance.

    
    # Parameters:
    - solid : 2D numpy array (Nx, Ny) of type bool
            True indicates a solid node.
    - T     : int
            Number of simulation time steps.
    - force_dir: int
            0 is x direction, 1 is y direction.
              
    # Returns:
    - u            : 3D numpy array (Nx, Ny, 2) velocity field.
    - kx            : Permeability in the x direction.
    - ky            : Permeability in the y direction.
    """
    Nx = solid.shape[0]
    Ny = solid.shape[1]
    
    # Create fluid mask: fluid = not solid:
    fluid = np.empty((Nx, Ny), dtype=np.bool_)
    for x in range(Nx):
        for y in range(Ny):
            if solid[x, y]:
                fluid[x, y] = False
            else:
                fluid[x, y] = True
    
    # Lattice vectors (9 directions):
    c = np.empty((9, 2), dtype=np.int64)
    c[0, 0] =  0; c[0, 1] =  0
    c[1, 0] =  1; c[1, 1] =  0
    c[2, 0] =  0; c[2, 1] =  1
    c[3, 0] = -1; c[3, 1] =  0
    c[4, 0] =  0; c[4, 1] = -1
    c[5, 0] =  1; c[5, 1] =  1
    c[6, 0] = -1; c[6, 1] =  1
    c[7, 0] = -1; c[7, 1] = -1
    c[8, 0] =  1; c[8, 1] = -1

    # Lattice weights:
    w = np.empty(9, dtype=np.float64)
    w[0] = 4.0/9.0
    w[1] = 1.0/9.0
    w[2] = 1.0/9.0
    w[3] = 1.0/9.0
    w[4] = 1.0/9.0
    w[5] = 1.0/36.0
    w[6] = 1.0/36.0
    w[7] = 1.0/36.0
    w[8] = 1.0/36.0

    # Bounce-back mapping:
    bounce_back_pairs = np.empty(9, dtype=np.int64)
    bounce_back_pairs[0] = 0
    bounce_back_pairs[1] = 3
    bounce_back_pairs[2] = 4
    bounce_back_pairs[3] = 1
    bounce_back_pairs[4] = 2
    bounce_back_pairs[5] = 7
    bounce_back_pairs[6] = 8
    bounce_back_pairs[7] = 5
    bounce_back_pairs[8] = 6

    # Initialize macroscopic variables:
    rho = np.empty((Nx, Ny), dtype=np.float64)
    u   = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                rho[x, y] = 1.0  # initial density

    # Gravity and forcing term:
    grav = 0.00001
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            F[x, y, force_dir] = -grav  # gravity in x-direction (adjust as needed)
            # F[x, y, 1] = 0.0

    # Relaxation parameter:
    omega = 0.6
    relax_corr = 1.0 - 1.0/(2.0 * omega)

    # Initialize lattice distributions f using equilibrium with forcing:
    f = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                # Square of velocity:
                u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                for i in range(9):
                    # Compute dot product u·c[i]:
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    # Forcing term contribution:
                    Fi = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
                    f[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi
    
    Fi = np.empty((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                for i in range(9):
                    Fi[x, y, i] = w[i] * relax_corr * 3.0 * (F[x, y, 0]*c[i, 0] + F[x, y, 1]*c[i, 1])
    # Main simulation loop
    for step in range(T):

        # Collision: compute equilibrium distribution and relax toward it
        feq = np.empty((Nx, Ny, 9), dtype=np.float64)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    u_sq = u[x, y, 0]*u[x, y, 0] + u[x, y, 1]*u[x, y, 1]
                    for i in range(9):
                        eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                        feq[x, y, i] = w[i]*rho[x, y]*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq) + Fi[x, y, i]
                        f[x, y, i] = f[x, y, i] + omega * (feq[x, y, i] - f[x, y, i])
        
        
        # Streaming step: propagate distributions
        f_stream = np.copy(f)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        if fluid[new_x, new_y]:
                            f_stream[new_x, new_y, i] = f[x, y, i]
                        else:
                            f_stream[x, y, bounce_back_pairs[i]] = f[x, y, i]
        f = f_stream  # update f
        
        # Update macroscopic variables: density and velocity
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x,y]:
                    sum_f = 0.0
                    u0 = 0.0
                    u1 = 0.0
                    for i in range(9):
                        sum_f += f[x, y, i]
                        u0 += f[x, y, i] * c[i, 0]
                        u1 += f[x, y, i] * c[i, 1]
                    rho[x, y] = sum_f
                    if sum_f != 0.0:
                        u[x, y, 0] = u0 / sum_f
                        u[x, y, 1] = u1 / sum_f
                    else:
                        u[x, y, 0] = 0.0
                        u[x, y, 1] = 0.0
        
    u_x = 0.0
    u_y = 0.0
    tot_rho = 0.0
    num = 0.0
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x,y]:
                u_x += u[x, y,0] 
                u_y += u[x, y,1] 
                tot_rho += rho[x, y]
                num += 1.0
    avg_u_x = u_x/num
    avg_u_y = u_y/num
    avg_rho = tot_rho/num
    mu = (relax_corr-1/2)/3
    k_x = avg_u_x*mu/(avg_rho*grav)
    k_y = avg_u_y*mu/(avg_rho*grav)
    return u, k_x, k_y