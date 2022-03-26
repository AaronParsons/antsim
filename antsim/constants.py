import numpy as np

MU0 = 4 * np.pi * 1e-7 # H/m, permeability of free space
C = 2.99792458e8 # m/s
EPS0 = 1. / (MU0 * C**2) # F/m, permittivity of free space
Z0 = MU0 * C # Ohms, impedance of free space

def courant(dimensions):
    """
    Return the default Courant number for an N-dimensiona simulation.
    Arguments:
        dimensions: int, number of dimensions in simulation
    """
    return 1 / np.sqrt(dimensions)
