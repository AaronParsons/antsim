'''An initial 1D test of FDTD.'''

import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt

SIZE = 200 # size of 1D grid
Z_fs = 377. # Ohms, impedance of free space

def run(steps, size=SIZE, plot_every=1):
    '''
    Run a 1D FDTD simulation and plot the results.
    
    Arguments:
        steps: int, number of time steps to run
        size: int, size of the FDTD grid. Default 200
        plot_every: int, step between plot updates
    '''
    hy = np.zeros(size)
    ez = np.zeros(size)

    # Add an impedance discontinuity
    # Slicing of Z has to follow H
    Z = np.ones(size) * Z_fs
    Z[-size//4+1:] *= 2

    # Initialize plotting
    plt.ion()
    fig = plt.figure()
    E_plot, = plt.plot(np.arange(size), ez, label='$E_z$')
    H_plot, = plt.plot(np.arange(size) + 0.5, hy, label='$H_y$')
    Z_plot, = plt.plot(np.arange(size), Z / Z_fs, label='$Z$')
    plt.ylim(-3,3)
    plt.legend()

    for t in tqdm.tqdm(range(steps)):
        # simple H absorbing boundary condition
        hy[-1] = hy[-2] # ABC for upper edge
        # first update magnetic field
        hy[:-1] += np.diff(ez) / Z[:-1]

        # update additive source
        hy[size//4-1] -= np.exp(-(t-30)**2 / 100) / Z[size//4-1]

        # simple E absorbing boundary condition
        ez[0] = ez[1]
        # next update electric field
        ez[1:] += np.diff(hy) * Z[:-1]
        # update additive source
        ez[size//4] += np.exp(-(t+1-30)**2 / 100)
        

        # update plots
        if t % plot_every == 0:
            E_plot.set_ydata(ez)
            H_plot.set_ydata(hy * Z)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

if __name__ == '__main__':
    run(450, plot_every=1)
