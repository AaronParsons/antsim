'''An initial 1D test of FDTD.'''

import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt

SIZE = 200 # size of 1D grid

# Constants
MU0 = 4 * np.pi * 1e-7 # H/m, permeability of free space
C = 2.99792458e8 # m/s
EPS0 = 1. / (MU0 * C**2) # F/m, permittivity of free space
Z0 = MU0 * C # Ohms, impedance of free space

class simulator:

    def __init__(self, steps, grid_size, sc=1.):
        self.grid_size=grid_size
        self.steps=steps
        self.sc=sc  # courant number

        # initialize grid, loss, fields, impedance...
        self.grid=np.zeros(self.grid_size)
        self.loss=np.zeros_like(self.grid)
        # Define Yee lattice
        # E0 H0 E1 H1 ... HN
        # so that, e.g., H0 can be regarded as being at E0.5       
        self.hy = np.zeros_like(self.grid)
        self.ez = np.zeros_like(self.grid)
        self.impedance=Z0*np.ones_like(self.grid)

    def add_loss(self, loss=0.02, thickness_ratio=0.5):
        thickness=int(thickness_ratio*self.grid_size)
        self.loss[:-thickness]=0
        self.loss[-thickness:]=loss


    def add_source(self):


    def update_fields(self):
        self.hy[:-1] += np.diff(ez) / self.impedance[:-1]
        self.ez[1:] += np.diff(hy) * self.impedance[:-1]

def run(steps, size=SIZE, plot_every=1):
    '''
    Run a 1D FDTD simulation and plot the results.
    
    Arguments:
        steps: int, number of time steps to run
        size: int, size of the FDTD grid. Default 200
        plot_every: int, step between plot updates
    '''
    # Define Yee lattice, which alternates:
    # E0 H0 E1 H1 ... HN-1 EN HN
    # so that, e.g
    hy = np.zeros(size)
    ez = np.zeros(size)

    # Add spatial impedance, slicing of Z follows H
    Z = np.ones(size) * Z0
    Z[-size//2:-size//4] = np.linspace(1, 2, size//4) * Z0
    Z[-size//4:] = 2 * Z0

    # Initialize plotting
    plt.ion()
    fig = plt.figure()
    E_plot, = plt.plot(np.arange(size), ez, label='$E_z$')
    H_plot, = plt.plot(np.arange(size) + 0.5, hy, label='$H_y$')
    Z_plot, = plt.plot(np.arange(size), Z / Z0, label='$Z$')
    plt.ylim(-3,3)
    plt.legend()

    src_x = 10

    # define edge before boundary conditions are imposed
    L_edge, R_edge = 1, -2
    P_out_r = 0 # holds integrated power off right edge
    P_out_l = 0 # holds integrated power off left edge

    # Main simulation loop
    for t in tqdm.tqdm(range(steps)):

        # impose H absorbing boundary condition
        hy[-1] = hy[-2] # ABC for upper edge

        # update H
        hy[:-1] += np.diff(ez) / Z[:-1]

        # update H of additive source
        hy[src_x-1] -= np.exp(-(t-30)**2 / 100) / Z[src_x-1]

        # impose E absorbing boundary condition
        ez[0] = ez[1]

        # update E
        ez[1:] += np.diff(hy) * Z[:-1]

        # update E of additive source
        ez[src_x] += np.exp(-(t+1-30)**2 / 100)
        
        # calculate quantities we are tracking vs. time
        # fluxes off left and right edges
        F_out_r = -0.5 * (hy[R_edge] + hy[R_edge-1]) * ez[R_edge] / MU0
        F_out_l = 0.5 * hy[L_edge] * (ez[L_edge] + ez[L_edge+1]) / MU0
        # accumulated power off left and right edges
        P_out_r += F_out_r
        P_out_l += F_out_l

        # update plots
        if t % plot_every == 0:
            E_plot.set_ydata(ez)
            H_plot.set_ydata(hy * Z)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

    # print summary report
    print(P_out_l, P_out_r, P_out_r+P_out_l)

if __name__ == '__main__':
    run(450, plot_every=10)
