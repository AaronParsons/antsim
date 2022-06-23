import numpy as np
from constants import Z0, courant


class Grid1D:
    def __init__(self, size, Z=Z0, courant=courant(1), dtype=np.float32):
        """
        Arguments:
            size: int, size of 1D grid
            Z: float or array, impedance. Default Z0.
            courant: Courant number of simulation. Default 1.
            dtype: float32 or float64, precision of simulation.
        """
        # simulation constants
        self.Z = Z * np.ones(size, dtype=dtype)  # ensure shape matches
        self.dtds = courant
        # fields define a Yee lattice, which alternates:
        # E0 H0 E1 H1 ... HN-1 EN
        # so that, e.g., H0 can be regarded as being at E0.5
        self.Hy = np.zeros(size, dtype=dtype)
        self.Ez = np.zeros(size, dtype=dtype)
        # propagation constants
        # self.C_EzE = np.ones(size  , dtype=dtype)
        # self.C_EzH = np.ones(size  , dtype=dtype)
        # self.C_EzH[:-1] *= self.dtds * self.Z
        # self.C_HyH = np.ones(size-1, dtype=dtype)
        # self.C_HyE = np.ones(size-1, dtype=dtype)
        # self.C_HyE *= self.dtds / self.Z

    def update_H(self):
        """Advance H-fields one time step."""
        self.Hy[:-1] += np.diff(self.Ez) / self.Z[:-1]

    def update_E(self):
        """Advance E-fields one time step."""
        self.Ez[1:] += np.diff(self.Hy) * self.Z[1:]

    def boundary_abc_E(self):
        """Impose an absorbing boundary condition (ABC) on E-field."""
        self.Ez[0] = self.Ez[1]

    def boundary_abc_H(self):
        """Impose an absorbing boundary condition (ABC) on H-field."""
        self.Hy[-1] = self.Hy[-2]
