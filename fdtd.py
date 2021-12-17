'''An initial 1D test of FDTD.'''

import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

try:
    import sources
    from constants import MU0, C, EPS0, Z0
except ImportError:
    from antsim import sources
    from antsim.constants import MU0, C, EPS0, Z0

class simulator:

    def __init__(self, grid_size, sc=1.):
        self.grid_size=grid_size
        self.steps=0
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

        self.poynting_flux=np.empty((0, 2))  # each row is time,firsdt col=left edge, 2nd col=right edge
 #       self.E_flux=np.empty((0,3))

    def add_loss(self, loss=0.02, thickness_ratio=0.5, right_edge=0, epsr=4.):
        thickness=int(thickness_ratio*self.grid_size)
        self.loss[:-(thickness+right_edge)]=0.
        if right_edge>0:
            self.loss[-(thickness+right_edge):-right_edge]=loss
            self.loss[-right_edge:]=0
        else:
            self.loss[-thickness:]=loss
        self.epsr[:-(thickness+right_edge)]=1.
        if right_edge>0:
            self.epsr[-(thickness+right_edge):-right_edge]=epsr
            self.epsr[-right_edge:]=1.
        else:
            self.epsr[-thickness:]=epsr

    def add_source(self, name, source_loc, **kwargs):
        if name=='harmonic':
            ppw=kwargs.pop('ppw', 1000)
            def source(m, q):
                return sources.harmonic(m, q, ppw, self.sc)
        elif name=='ricker':
            N_p=kwargs.pop('ppw', 5000)
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

#    def collect_E_flux(self, timestep, ppw, left_edge=1, right_edge=-2):
#        fr=self.ez[right_edge]
#        fl=self.ez[left_edge]
#        s=sources.harmonic(-.5, timestep+.5, ppw, 1)  # hardcoded for now...
#        self.E_flux[timestep]=[s, fl, fr]


    def run(self, time_steps, plot_every, source_fcn='harmonic', source_loc=10, **kwargs):
        self.steps=time_steps
        self.poynting_flux=np.empty((self.steps, 2))
        source=self.add_source(source_fcn, source_loc, **kwargs)
        # create figure to plot
        plt.ion()
        fig=plt.figure()
        E_plot,=plt.plot(np.arange(self.grid_size), self.ez, label='$E_z$')
        H_plot,=plt.plot(np.arange(self.grid_size)+0.5, self.hy, label='$H_y$')
        if len(np.nonzero(self.loss)[0])>0:
            plt.axvspan(xmin=np.nonzero(self.loss)[0][0], xmax=np.nonzero(self.loss)[0][-1], ymin=-Z0, ymax=Z0, alpha=.5)
        elif len(np.nonzero(self.epsr-1)[0])>0:
            plt.axvspan(xmin=np.nonzero(self.epsr-1)[0][0], xmax=np.nonzero(self.epsr-1)[0][-1], ymin=-Z0, ymax=Z0,
                    alpha=.3)
        plt.ylim(-Z0, Z0)
        #plt.xlim()
        plt.legend(loc='upper left')

        for q in tqdm(range(self.steps)):
            self.update_fields(q, source)
            self.calc_flux(q)
            if q%plot_every==0:
                E_plot.set_ydata(self.ez)
                H_plot.set_ydata(self.hy*Z0)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(.1)

                

    def plot_spectrum(self):
        left=self.poynting_flux[:, 0]**2  # power
        right=self.poynting_flux[:, 1]**2
        freqs=np.fft.rfftfreq(len(left)).real # DFT sample freqs
        spec_l=np.fft.rfft(left).real
        spec_r=np.fft.rfft(right).real
        # plot
        plt.figure()
        plt.plot(freqs, spec_l/np.max(np.abs(spec_l)), label='outgoing left')
        plt.plot(freqs, spec_r/np.max(np.abs(spec_r)), label='outgoing right')
        plt.ylim(-1.2,1.2)
      #  plt.xlim(0,freqs[50])
        plt.legend()
        plt.show()
        plt.pause(0.01)
        input('Hit Enter')

if __name__=='__main__':
    test=simulator(1000)
    test.add_loss(loss=0.02, thickness_ratio=0.1, epsr=4)
    test.run(2000-1, 10, 'harmonic', 500, ppw=5)
    print('Total power out = {}'.format(np.sum(self.poynting_flux[:,-1]**2)))
    #test=simulator(10000)
    #test.add_loss(loss=0.01, thickness_ratio=0.1, epsr=1)
    #test.run(100000, 1000, 'ricker', 10, ppw=500, delay_multiple=1)
    test.plot_spectrum()
    test.plot_spectrum(spectrum='poynting')
