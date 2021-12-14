'''An initial 1D test of FDTD.'''

import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import sources


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
        self.epsr=np.ones_like(self.grid)

        self.source_loc=0

        self.poynting_flux=np.empty((self.steps, 2))  # each row is time,firsdt col=left edge, 2nd col=right edge

    def add_loss(self, loss=0.02, thickness_ratio=0.5, epsr=4.):
        thickness=int(thickness_ratio*self.grid_size)
        self.loss[:-thickness]=0.
        self.loss[-thickness:]=loss
        self.epsr[:-thickness]=1.
        self.epsr[-thickness:]=epsr  # epsilon/espilon0

    def add_source(self, name, source_loc, **kwargs):
        if name=='harmonic':
            ppw=kwargs.pop('ppw', 1000)
            def source(m, q):
                return sources.harmonic(m, q, ppw, self.sc)
        elif name=='ricker':
            N_p=kwargs.pop('peak_frequency', 1000)
            M_d=kwargs.pop('delay_multiple', 1)
            def source(m, q):
                return sources.ricker(m, q, N_p, M_d, self.sc)
        else:
            print('Invalid source name, must be "harmonic" or "ricker"')
            return None
        self.source_loc=source_loc
        return source


    def update_fields(self, timestep, source):
        """
        Only electric loss, no magnetic
        """
        self.hy[-1]=self.hy[-2]  # ABC
        self.hy[:-1]+=np.diff(self.ez)/self.impedance[:-1]
        if source is not None:
            self.hy[self.source_loc-1]-=source(0, timestep)
        self.ez[0]=self.ez[1]  # ABC
        self.ez[1:]*=(1.-self.loss[1:])/(1.+self.loss[1:]) 
        self.ez[1:]+=np.diff(self.hy)*self.impedance[1:]/(self.epsr[1:]*(1+self.loss[1:]))  # den=1 for vacuum
        if source is not None:
            self.ez[self.source_loc]+=source(-.5, timestep+.5)
   
    def calc_flux(self, timestep, left_edge=1, right_edge=-2):
        flux_r=-self.ez[right_edge]*(self.hy[right_edge]+self.hy[right_edge-1])/2
        flux_r/=MU0
        flux_l=self.hy[left_edge]*(self.ez[left_edge]+self.ez[right_edge+1])/2
        flux_l/=MU0
        self.poynting_flux[timestep]=[flux_l, flux_r]


    def run(self, plot_every, source_fcn=None, source_loc=10, **kwargs):
        if source_fcn is not None:
            source=self.add_source(source_fcn, source_loc, **kwargs)
        else:
            source=None
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
            self.update_fields(q, source)
            self.calc_flux(q)
            if q%plot_every==0:
                E_plot.set_ydata(self.ez)
                H_plot.set_ydata(self.hy*Z0)  # prob wrong
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(.1)
                

    def plot_spectrum(self):
        plt.figure()
        left=self.poynting_flux[:, 0]
        right=self.poynting_flux[:, 1]
        spec_l=np.fft.rfft(left)
        spec_r=np.fft.rfft(right)
        plt.plot(spec_l/np.max(np.abs(spec_l)), label='incoming')
        plt.plot(spec_r/np.max(np.abs(spec_r)), label='outgoing')
        plt.ylim(-1,1)
        plt.xlim(0,1500)
        plt.legend()
        plt.show()
        plt.pause(0.01)
        input("<Hit Enter to close>")

#test=simulator(10000, 1000)
#test.add_loss()
#test.run(50, 'harmonic', 10)
test=simulator(3000, 1000)
test.add_loss()
test.run(50, 'ricker')
test.plot_spectrum()

