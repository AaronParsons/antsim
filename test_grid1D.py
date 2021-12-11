"""Test Grid1D objects and methods."""
import numpy as np
import pytest

import grid1D

SIZE = 50

def test_init():
    """Make sure Grid1D object initializes sensibly."""
    g = grid1D.Grid1D(SIZE)
    assert np.all(g.Z == grid1D.Z0) # defaults to free-space
    assert g.dtds == 1 # courant number
    assert g.Hy.size == SIZE 
    assert g.Hy.dtype == np.float32
    assert g.Ez.size == SIZE
    assert g.Ez.dtype == np.float32
    assert np.all(g.Ez == 0)
    assert np.all(g.Hy == 0)
    g = grid1D.Grid1D(SIZE, dtype=np.float64)
    assert g.Hy.dtype == np.float64
    assert g.Ez.dtype == np.float64
    with pytest.raises(ValueError):
        Z = np.ones(SIZE//2)
        g = grid1D.Grid1D(SIZE, Z=Z)

def test_update_H():
    """Make sure Grid1D update H field is numerically correct."""
    g = grid1D.Grid1D(SIZE)
    g.update_H()
    assert np.all(g.Hy == 0)
    g.Ez[SIZE//2] = 1
    g.update_H()
    assert np.all(g.Hy[:SIZE//2-1] == 0)
    assert np.allclose(g.Hy[SIZE//2-1], 1/grid1D.Z0, atol=1e-7)
    assert np.allclose(g.Hy[SIZE//2], -1/grid1D.Z0, atol=1e-7)
    assert np.all(g.Hy[SIZE//2+1:] == 0)

def test_update_E():
    """Make sure Grid1D update E field is numerically correct."""
    g = grid1D.Grid1D(SIZE)
    g.update_E()
    assert np.all(g.Ez == 0)
    g.Hy[SIZE//2] = 1
    g.update_E()
    assert np.all(g.Ez[:SIZE//2] == 0)
    assert np.allclose(g.Ez[SIZE//2], grid1D.Z0, atol=1e-7)
    assert np.allclose(g.Ez[SIZE//2+1], -grid1D.Z0, atol=1e-7)
    assert np.all(g.Ez[SIZE//2+2:] == 0)

def test_abc():
    """Run basic simulation and make sure absorbing boundary conditions
    eliminate power from inside simulation."""
    g = grid1D.Grid1D(SIZE)
    for t in range(75):
        g.boundary_abc_H()
        g.update_H()
        g.boundary_abc_E()
        g.update_E()
        g.Ez[SIZE//2] += np.exp(-(t+1-20)**2 / (2*5**2))
    # Show that input power is attenuated away
    assert np.all(np.abs(g.Ez)**2 < 1e-7)
    assert np.all(np.abs(g.Hy * g.Z)**2 < 1e-7)
