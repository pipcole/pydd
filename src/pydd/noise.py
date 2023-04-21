try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
from . import noise_resources

"""
Noise PSDs.
"""


f_range_LISA = (2e-5, 1.0)  # Hz, from LISA Science Requirements fig. 1

def S_n_LISA(f):
    """
    LISA noise PSD, averaged over sky position and polarization angle.

    Reference:
        Travis Robson et al 2019 Class. Quantum Grav. 36 105011
        https://arxiv.org/abs/1803.01944
    """
    return np.where(
        np.logical_and(f >= f_range_LISA[0], f <= f_range_LISA[1]),
        1
        / f ** 14
        * 1.80654e-17
        * (0.000606151 + f ** 2)
        * (3.6864e-76 + 3.6e-37 * f ** 8 + 2.25e-30 * f ** 10 + 8.66941e-21 * f ** 14),
        np.inf,
    )


def load_S_n(name):
    #quick fix - not sure what happened with the path
    path_context = pkg_resources.path(noise_resources, f"{name}.dat")
    with path_context as path:
        _fs, _sqrt_S_ns = np.loadtxt(path, unpack=True)
    _fs, _sqrt_S_ns = np.loadtxt(path, unpack=True)
    S_n = lambda f: np.interp(f, _fs, _sqrt_S_ns ** 2, np.inf, np.inf)
    f_range = (_fs[0], _fs[-1])
    return S_n, f_range


S_n_aLIGO, f_range_aLIGO = load_S_n("aLIGO")
S_n_ce, f_range_ce = load_S_n("ce")
S_n_et, f_range_et = load_S_n("et")
