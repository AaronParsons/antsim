import numpy as np

def harmonic(m, q, ppw, sc):
    """
    m = float: spatial step
    q = float: temporal step
    sc = float: courant number
    ppw = float: points per wavelength
    """
    arg=2*np.pi/ppw*(sc*q-m)
    return np.sin(arg)

def ricker(m, q, N_p, M_d, sc):
    """
    N_p = float: ppw at peak frequency
    M_d = float: delay multiple
    """
    arg=(sc*q-m)/N_p-M_d
    arg*=np.pi
    prefactor=1-2*arg**2
    return prefactor*np.exp(-arg**2)

