import numpy as np


def harmonic(m, q, ppw, sc):
    """
    Hamornic source.

    Parameters
    ----------
    m : float
        Spatial step.
    q : float
        Temporal step.
    ppw : float
        Points per wavelength.
    sc : float
        Courant number.

    Returns
    -------
    source : float
        The harmonic source evaluated at the specified space and time.

    """
    arg = 2 * np.pi / ppw * (sc * q - m)
    source = np.sin(arg)
    return source


def ricker(m, q, N_p, M_d, sc):
    """
    N_p = float: ppw at peak frequency
    M_d = float: delay multiple
    """
    arg = (sc * q - m) / N_p - M_d
    arg *= np.pi
    prefactor = 1 - 2 * arg**2
    return prefactor * np.exp(-(arg**2))
