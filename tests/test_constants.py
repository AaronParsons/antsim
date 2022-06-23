import numpy as np
import pytest

import constants

def test_values():
    assert np.allclose(constants.Z0, 376.730, atol=1e-3)

def test_courant():
    assert constants.courant(1) == 1
    assert np.allclose(constants.courant(2), 2**-0.5, atol=1e-7)
    assert np.allclose(constants.courant(3), 3**-0.5, atol=1e-7)
