'''An initial 1D test of FDTD.'''

import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from sources import harmonic


# Constants
MU0 = 4 * np.pi * 1e-7 # H/m, permeability of free space
C = 2.99792458e8 # m/s
EPS0 = 1. / (MU0 * C**2) # F/m, permittivity of free space
Z0 = MU0 * C # Ohms, impedance of free space

class simulator:

    def __init__(self, steps, grid_size, sc=1., source_bd=10, source_ppw=1000):
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
        self.epsr=np.ones_like(self.grid)

        # TFSF boundary
        self.source_bd=source_bd
        self.source_ppw=source_ppw

    def add_loss(self, loss=0.02, thickness_ratio=0.5, epsr=4.):
        thickness=int(thickness_ratio*self.grid_size)
        self.loss[:-thickness]=0.
        self.loss[-thickness:]=loss
        self.epsr[:-thickness]=1.
        self.epsr[-thickness:]=epsr  # epsilon/espilon0

    def update_fields(self, timestep):
        """
        Only electric loss, no magnetic
        """
        self.hy[-1]=self.hy[-2]  # ABC
        self.hy[:-1]+=np.diff(self.ez)/self.impedance[:-1]
        self.hy[self.source_bd-1]-=harmonic(0, timestep, self.sc, self.source_ppw)
        self.ez[0]=self.ez[1]  # ABC
        self.ez[1:]*=(1.-self.loss[1:])/(1.+self.loss[1:]) 
        self.ez[1:]+=np.diff(self.hy)*self.impedance[1:]/(self.epsr[1:]*(1+self.loss[1:]))  # den=1 for vacuum
        self.ez[self.source_bd]+=harmonic(-.5, timestep+.5, self.sc, self.source_ppw)
   
    def run(self, plot_every, source_ppw=None):
        if source_ppw:
            self.source_ppw=source_ppw

        # create figure to plot
        plt.ion()
        fig=plt.figure()
        E_plot,=plt.plot(np.arange(self.grid_size), self.ez, label='$E_z$')
        H_plot,=plt.plot(np.arange(self.grid_size)+0.5, self.hy, label='$H_y$')
        #atm_plot,=plt.plot(self.loss, label='Atmosphere')
        plt.ylim(-Z0, Z0)
        #plt.xlim()
        plt.legend()

        for q in tqdm(range(self.steps)):
            self.update_fields(q)
            if q%plot_every==0:
                E_plot.set_ydata(self.ez)
                H_plot.set_ydata(self.hy*Z0)  # prob wrong
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(.1)
                

test=simulator(10000, 1000)
#test.add_loss()
test.run(50)


