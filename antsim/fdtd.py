import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import warnings

from antsim.constants import MU0, C, EPS0, Z0
import antsim.sources


class Simulator:
    def __init__(self, grid_size, sc=1.0):
        self.grid_size = grid_size
        self.steps = 0
        self.sc = sc  # courant number

        # initialize grid, loss, fields, impedance...
        self.grid = np.zeros(self.grid_size)
        self.loss = np.zeros_like(self.grid)
        # Define Yee lattice
        # E0 H0 E1 H1 ... HN
        # so that, e.g., H0 can be regarded as being at E0.5
        self.hy = np.zeros_like(self.grid)
        self.ez = np.zeros_like(self.grid)
        self.impedance = Z0 * np.ones_like(self.grid)
        self.epsr = np.ones_like(self.grid)

        self.source_loc = 0
        self.source = lambda x : 0

    def add_loss(self, loss=0.02, thickness_ratio=0.5, right_edge=0, epsr=4.0):
        thickness = int(thickness_ratio * self.grid_size)
        self.loss[: -(thickness + right_edge)] = 0.0
        if right_edge > 0:
            self.loss[-(thickness + right_edge) : -right_edge] = loss
            self.loss[-right_edge:] = 0
        else:
            self.loss[-thickness:] = loss
        self.epsr[: -(thickness + right_edge)] = 1.0
        if right_edge > 0:
            self.epsr[-(thickness + right_edge) : -right_edge] = epsr
            self.epsr[-right_edge:] = 1.0
        else:
            self.epsr[-thickness:] = epsr

    def add_source(self, name, source_loc, **kwargs):
        if name == "harmonic":
            ppw = kwargs.pop("ppw", 1000)  # points per wavelength

            def source(m, q):
                return sources.harmonic(m, q, ppw, self.sc)

        elif name == "ricker":
            N_p = kwargs.pop("ppw", 5000)  # points per wavelength
            M_d = kwargs.pop("delay_multiple", 1)  # delay multiple

            def source(m, q):
                return sources.ricker(m, q, N_p, M_d, self.sc)

        else:
            warning.warn(
                "Invalid source name, must be 'harmonic' or 'ricker', not"
                f"{name}.",
                UserWarning
            )
            return None
        self.source_loc = source_loc
        self.source = source

    def update_fields(self, timestep, source):
        """
        Only electric loss, no magnetic
        """
        self.hy[-1] = self.hy[-2]  # ABC
        self.hy[:-1] += np.diff(self.ez) / self.impedance[:-1]
        if source is not None:
            self.hy[self.source_loc - 1] -= source(0, timestep)
        self.ez[0] = self.ez[1]  # ABC
        self.ez[1:] *= (1.0 - self.loss[1:]) / (1.0 + self.loss[1:])
        self.ez[1:] += (
            np.diff(self.hy)
            * self.impedance[1:]
            / (self.epsr[1:] * (1 + self.loss[1:]))
        )  # the denominator is 1 for vacuum
        if source is not None:
            self.ez[self.source_loc] += source(-0.5, timestep + 0.5)

    def run(
        self,
        time_steps,
        plot_every,
    ):
        self.steps = time_steps
        # create figure to plot
        plt.ion()
        fig = plt.figure()
        (E_plot,) = plt.plot(np.arange(self.grid_size), self.ez, label="$E_z$")
        (H_plot,) = plt.plot(
            np.arange(self.grid_size) + 0.5, self.hy, label="$H_y$"
        )
        if len(np.nonzero(self.loss)[0]) > 0:
            plt.axvspan(
                xmin=np.nonzero(self.loss)[0][0],
                xmax=np.nonzero(self.loss)[0][-1],
                ymin=-Z0,
                ymax=Z0,
                alpha=0.5,
            )
        elif len(np.nonzero(self.epsr - 1)[0]) > 0:
            plt.axvspan(
                xmin=np.nonzero(self.epsr - 1)[0][0],
                xmax=np.nonzero(self.epsr - 1)[0][-1],
                ymin=-Z0,
                ymax=Z0,
                alpha=0.3,
            )
        plt.ylim(-Z0, Z0)
        # plt.xlim()
        plt.legend(loc="upper left")

        for q in tqdm(range(self.steps)):
            self.update_fields(q, self.source)
            self.calc_flux(q)
            if q % plot_every == 0:
                E_plot.set_ydata(self.ez)
                H_plot.set_ydata(self.hy * Z0)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.1)

if __name__ == "__main__":
    sim = simulator(1000)
    sim.add_loss(loss=0, thickness_ratio=0.1, epsr=4)
    sim.add_source("harmonic", 500, ppw=10)
    sim.run(2000 - 1, 10, "harmonic", 500, ppw=10)
