"""An initial 1D test of FDTD."""

import numpy as np
import tqdm
import time
import matplotlib.pyplot as plt
from constants import Z0
from grid1D import Grid1D

SIZE = 200  # size of 1D grid


def run(steps, size=SIZE, plot_every=1):
    """
    Run a 1D FDTD simulation and plot the results.

    Arguments:
        steps: int, number of time steps to run
        size: int, size of the FDTD grid. Default 200
        plot_every: int, step between plot updates
    """
    Z = np.ones(size) * Z0
    # Add spatial impedance, slicing of Z follows H
    # Z[-size//2:-size//4] = np.linspace(1, 2, size//4) * Z0
    # Z[-size//4:] = 2 * Z0
    g = Grid1D(SIZE, Z=Z)

    # Initialize plotting
    plt.ion()
    fig = plt.figure()
    (E_plot,) = plt.plot(np.arange(g.Ez.size), g.Ez, label="$E_z$")
    (H_plot,) = plt.plot(np.arange(g.Hy.size) + 0.5, g.Hy, label="$H_y$")
    (Z_plot,) = plt.plot(np.arange(g.Z.size), g.Z / Z0, label="$Z$")
    plt.ylim(-3, 3)
    plt.legend()

    src_x = SIZE // 2

    # define edge before boundary conditions are imposed
    L_edge, R_edge = 1, -2
    P_out_r = 0  # holds integrated power off right edge
    P_out_l = 0  # holds integrated power off left edge

    # Main simulation loop
    for t in tqdm.tqdm(range(steps)):
        g.boundary_abc_H()
        g.update_H()

        # update H of additive source
        # g.Hy[src_x-1] -= np.exp(-(t-30)**2 / 100) / g.Z[src_x-1]

        g.boundary_abc_E()
        g.update_E()

        # update E of additive source
        # g.Ez[src_x] += np.exp(-(t+1-30)**2 / 100)
        g.Ez[src_x] += np.exp(-((t + 1 - 20) ** 2) / (2 * 5**2))

        # update plots
        if t % plot_every == 0:
            E_plot.set_ydata(g.Ez)
            H_plot.set_ydata(g.Hy * g.Z)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

    # print summary report
    # print(P_out_l, P_out_r, P_out_r+P_out_l)


if __name__ == "__main__":
    run(300, plot_every=2)
