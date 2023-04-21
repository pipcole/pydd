from math import pi
from typing import Callable, NamedTuple, Tuple, Type, Union
from scipy.optimize import minimize_scalar

import numpy as np
from scipy.special import betainc

from scipy.interpolate import interp2d, RegularGridInterpolator
from scipy.special import hyp2f1
from pydd.gatom import gatom_interp, R_211, ion_r_co_211, ion_E_co_211

#from CDMsur_Backbone import *
#from utils_surr import *
#from dill import load
#model = load(open("cdm_surr_new_data_1000.pkl", "rb")) # load trained surrogate model

"""
Functions for computing waveforms and various parameters for different types of
binaries.

While we use different notation in our paper, the `_to_c` notation means that
the function returns zero when the input frequency equals the binary's
coalescence frequency. This makes it much easier to compute dephasings and the
like since phases don't need to be aligned manually.

Uses SI units.
"""

G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s


class VacuumBinary(NamedTuple):
    """
    GR-in-vacuum binary.
    """
# option to include PN corrections to GW emission term
 #   PN: np.ndarray
    M_chirp: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray


class StaticDress(NamedTuple):
    """
    A dark dress with a non-evolving DM halo.
    """

  #  PN: np.ndarray
    gamma_s: np.ndarray
    c_f: np.ndarray
    M_chirp: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray


class DynamicDress(NamedTuple):
    """
    A dark dress with an evolving DM halo.
    """

 #   PN: np.ndarray
    gamma_s: np.ndarray
    rho_6: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class AccretionDisk(NamedTuple):
    """
    Analytic approximation to thin accretion disk.

    """
#    PN: np.ndarray
    SigM2: np.ndarray
    inout: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class GravAtom(NamedTuple):
    """
    Numerical interpolation for GravAtom system with n=2, l=1, m=1 (various options editable in function)

    """

#    PN: np.ndarray
    alpha: np.ndarray
    epsilon_init: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class SurrogateDD(NamedTuple):

    """
    Surrogate model for dark dress with an evolving DM halo.
    """

#    PN: np.ndarray
    gamma_s: np.ndarray
    rho_6: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class DynamicDressShift(NamedTuple):

    """
    Surrogate model for dark dress with an evolving DM halo.
    """

    ig: np.ndarray
    gamma_s: np.ndarray
    rho_6: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class AccretionDiskShift(NamedTuple):
    """
    Analytic approximation to thin accretion disk.

    """
    ig: np.ndarray
    SigM2: np.ndarray
    inout: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class GravAtomShift(NamedTuple):
    """
    Numerical interpolation for GravAtom system with n=2, l=1, m=1 (various options editable in function)

    """

    ig: np.ndarray
    alpha: np.ndarray
    epsilon_init: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

class Combo(NamedTuple):
    """
    A combination of 2 or more systems from dynamic dress, accretion disk and gravitational atom.
    """

 #   PN: np.ndarray
    gamma_s: np.ndarray
    rho_6: np.ndarray
    SigM2: np.ndarray
    inout: np.ndarray
    alpha: np.ndarray
    epsilon_init: np.ndarray
    M_chirp: np.ndarray
    q: np.ndarray
    Phi_c: np.ndarray
    tT_c: np.ndarray
    dL: np.ndarray
    f_c: np.ndarray

Binary = Union[VacuumBinary, StaticDress, DynamicDress, AccretionDisk, GravAtom, SurrogateDD, DynamicDressShift, AccretionDiskShift, GravAtomShift, Combo]


#@jit
def get_M_chirp(m_1, m_2):
    return (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)


#@jit
def get_m_1(M_chirp, q):
    return (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp


#@jit
def get_m_2(M_chirp, q):
    return (1 + q) ** (1 / 5) * q ** (2 / 5) * M_chirp


#@jit
def get_r_isco(m_1):
    return 6 * G * m_1 / C ** 2


#@jit
def get_f_isco(m_1):
    return np.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


#@jit
def get_r_s(m_1, rho_s, gamma_s):
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)


#@jit
def get_rho_s(rho_6, m_1, gamma_s):
    a = 0.2
    r_6 = 1e-6 * PC
    m_tilde = ((3 - gamma_s) * a ** (3 - gamma_s)) * m_1 / (2 * np.pi)
    return (rho_6 * r_6 ** gamma_s / (m_tilde ** (gamma_s / 3))) ** (
        1 / (1 - gamma_s / 3)
    )


#@jit
def get_rho_6(rho_s, m_1, gamma_s):
    a = 0.2
    r_s = ((3 - gamma_s) * a ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)
    r_6 = 1e-6 * PC
    return rho_s * (r_6 / r_s) ** -gamma_s


#@jit
def get_xi(gamma_s):
    # Could use that I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)


#@jit
def get_c_f(m_1, m_2, rho_s, gamma_s):
    Lambda = np.sqrt(m_1 / m_2)
    M = m_1 + m_2
    c_gw = 64 * G ** 3 * M * m_1 * m_2 / (5 * C ** 5)
    c_df = (
        8
        * pi
        * np.sqrt(G)
        * (m_2 / m_1)
        * np.log(Lambda)
        * (rho_s * get_r_s(m_1, rho_s, gamma_s) ** gamma_s / np.sqrt(M))
        * get_xi(gamma_s)
    )
    return c_df / c_gw * (G * M / pi ** 2) ** ((11 - 2 * gamma_s) / 6)


#@jit
def get_f_eq(gamma_s, c_f):
    return c_f ** (3 / (11 - 2 * gamma_s))


#@jit
def get_a_v(M_chirp):
    return 1 / 16 * (C ** 3 / (pi * G * M_chirp)) ** (5 / 3)


#@jit
def PhiT(f, params: Binary):
    return 2 * pi * f * t_to_c(f, params) - Phi_to_c(f, params)


#@jit
def Psi(f, params: Binary):
    return 2 * pi * f * params.tT_c - params.Phi_c - pi / 4 - PhiT(f, params)


#@jit
def h_0(f, params: Binary):
    return np.where(
        f <= params.f_c,
        1
        / 2
        * 4
        * pi ** (2 / 3)
        * (G * params.M_chirp) ** (5 / 3)
        * f ** (2 / 3)
        / C ** 4
        * np.sqrt(2 * pi / abs(d2Phi_dt2(f, params))),
        0.0,
    )


#@jit
def amp(f, params: Binary):
    """
    Amplitude averaged over inclination angle.
    """
    return np.sqrt(4 / 5) * h_0(f, params) / params.dL


#@jit
def Phi_to_c(f, params: Binary):
    if isinstance(params, GravAtom):
        pga = master_call(f, params)[0]
        return pga(f) - pga(params.f_c)
    if isinstance(params, GravAtomShift):
        pga = master_call_shift(f, params)[0]
        return params.ig * (pga(f) - pga(params.f_c))
    else:
        return _Phi_to_c_indef(f, params) - _Phi_to_c_indef(params.f_c, params)


#@jit
def t_to_c(f, params: Binary):
    if isinstance(params, GravAtom):
        tga = master_call(f, params)[1]
        return tga(f) - tga(params.f_c)
    if isinstance(params, GravAtomShift):
        tga = master_call_shift(f, params)[1]
        return tga(f) - tga(params.f_c)
    else:
        return _t_to_c_indef(f, params) - _t_to_c_indef(params.f_c, params)


#@jit
def _Phi_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _Phi_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _Phi_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _Phi_to_c_indef_d(f, params)
    elif isinstance(params, AccretionDisk):
        return _Phi_to_c_indef_a(f, params)
    elif isinstance(params, GravAtom):
        return master_call(f, params)[0]
    elif isinstance(params, SurrogateDD):
        return _Phi_to_c_indef_sur(f, params)
    elif isinstance(params, DynamicDressShift):
        return _Phi_to_c_indef_ds(f, params)
    elif isinstance(params, AccretionDiskShift):
        return _Phi_to_c_indef_as(f, params)
    elif isinstance(params, GravAtomShift):
        return master_call_shift(f, params)[0]
    elif isinstance(params, Combo):
        return _Phi_to_c_indef_c(f, params)
    else:
        raise ValueError("unrecognized type")


#@jit
def _t_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _t_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _t_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _t_to_c_indef_d(f, params)
    elif isinstance(params, AccretionDisk):
        return _t_to_c_indef_a(f, params)
    elif isinstance(params, GravAtom):
        return master_call(f, params)[1]
    elif isinstance(params, SurrogateDD):
        return _t_to_c_indef_sur(f, params)
    elif isinstance(params, DynamicDressShift):
        return _t_to_c_indef_ds(f, params)
    elif isinstance(params, AccretionDiskShift):
        return _t_to_c_indef_as(f, params)
    elif isinstance(params, GravAtomShift):
        return master_call_shift(f, params)[1]
    elif isinstance(params, Combo):
        return _t_to_c_indef_c(f, params)
    else:
        raise ValueError("'params' type is not supported")


#@jit
def d2Phi_dt2(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return d2Phi_dt2_v(f, params)
    elif isinstance(params, StaticDress):
        return d2Phi_dt2_s(f, params)
    elif isinstance(params, DynamicDress):
        return d2Phi_dt2_d(f, params)
    elif isinstance(params, AccretionDisk):
        return d2Phi_dt2_a(f, params)
    elif isinstance(params, GravAtom):
        return master_call(f, params)[2](f)
    elif isinstance(params, SurrogateDD):
        return d2Phi_dt2_sur(f, params)
    elif isinstance(params, DynamicDressShift):
        return d2Phi_dt2_ds(f, params)
    elif isinstance(params, AccretionDiskShift):
        return d2Phi_dt2_as(f, params)
    elif isinstance(params, GravAtomShift):
        return params.ig * master_call_shift(f, params)[2](f)  
    elif isinstance(params, Combo):
        return d2Phi_dt2_c(f, params)  
    else:
        raise ValueError("'params' type is not supported")

# PN corrections if using, need to de-comment terms in each phase/time/derivative of phase function

def phasePN(nu, v):
    return (3715/1008 + 55 * nu /12) * v**2 - 10 * pi * v**3 +(15293365/1016064 + 27145 * nu /1008 + 3085 * nu**2 / 144) * v**4

def tPN(nu, v):
    return (743/252 + 11 * nu/3) * v**2 - (32 * pi * v**3)/5 + (3058673/508032 + 5429 * nu/504 + (617* nu**2) /72) * v**4

def ddphasePN(f, m1, m2, nu):
    return (-195084288*C**3*f**(11/3)*
        ((f*G*(m1 + m2))/C**3)**(1/3)*pi**(11/3))/  (5*((C**3*(m1 + m2)**0.2)/(G*(m1*m2)**0.6))**(5/3)*
        (3024*f*G*(m1 + m2)*(743 + 924*nu)*pi**(2/3) + 
        ((f*G*(m1 + m2))/C**3)**(1/3)*
        (1016064*C**3 + f*G*(m1 + m2)*
        (3058673*((f*G*(m1 + m2))/C**3)**(1/3) + 
        1008*((f*G*(m1 + m2))/C**3)**(1/3)*nu*(5429 + 4319*nu) - 
        4064256*pi**(2/3))*pi**(4/3))))

# Vacuum binary

def _Phi_to_c_indef_v(f, params: VacuumBinary):
    m_1 = get_m_1(params.M_chirp, 1e-4)
    m_2 = get_m_2(params.M_chirp, 1e-4)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    v1 = ((G/C**3) * np.pi * (m_1 + m_2) * f)**(1/3)
    return get_a_v(params.M_chirp) / f ** (5 / 3)# * (1 + params.PN * phasePN(nu1, v1))


def _t_to_c_indef_v(f, params: VacuumBinary):
    m_1 = get_m_1(params.M_chirp, 1e-4)
    m_2 = get_m_2(params.M_chirp, 1e-4)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    v1 = ((G/C**3) * np.pi * (m_1 + m_2) * f)**(1/3)
    return 5 * get_a_v(params.M_chirp) / (16 * pi * f ** (8 / 3))# *(1 + params.PN * tPN(nu1, v1))



def d2Phi_dt2_v(f, params: VacuumBinary):
    m_1 = get_m_1(params.M_chirp, 1e-4)
    m_2 = get_m_2(params.M_chirp, 1e-4)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    newton = 12 * pi ** 2 * f ** (11 / 3) / (5 * get_a_v(params.M_chirp))
    return newton# + params.PN * (-ddphasePN(f, m_1, m_2, nu1) - newton)



def make_vacuum_binary(
    m_1,
    m_2,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> VacuumBinary:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return VacuumBinary(M_chirp, Phi_c, tT_c, dL, f_c)


# Interpolator for special version of hypergeometric function
def hypgeom_scipy(b, z):
    return hyp2f1(1, b, 1 + b, z)


def get_hypgeom_interps(n_bs=5000, n_zs=4950):
    bs = np.linspace(0.5, 1.99, n_bs)
    log10_abs_zs = np.linspace(-8, 6, n_zs)
    zs = -(10 ** log10_abs_zs)
    b_mg, z_mg = np.meshgrid(bs, zs, indexing="ij")


    vals_pos = np.array(hypgeom_scipy(b_mg, z_mg))
    vals_neg = np.log10(1 - hypgeom_scipy(-b_mg[::-1, :], z_mg))


    interp_pos = RegularGridInterpolator(
        (bs, log10_abs_zs), vals_pos
    )
    interp_neg = RegularGridInterpolator(
        (-bs[::-1], log10_abs_zs), 1 - 10 ** vals_neg
    )

    return interp_pos, interp_neg


interp_pos, interp_neg = get_hypgeom_interps()


def restricted_hypgeom(b, z: np.ndarray) -> np.ndarray:
    # Assumes b is a scalar
    if b>0:
        return interp_pos((b, np.log10(-z)))
    else:
        return interp_neg((b, np.log10(-z)))



def hypgeom_jax(b, z: np.ndarray) -> np.ndarray:
    # print(
    #     f"b: {b}, "
    #     f"log10(|z|) min: {jnp.log10(jnp.abs(z)).min()}, "
    #     f"log10(|z|) max: {jnp.log10(jnp.abs(z)).max()}"
    # )
    if b==1:
        return np.log(1 - z) / (-z)
    else:
        return restricted_hypgeom(b, z)

    #jax.lax.cond(
    #    b == 1, lambda z: np.log(1 - z) / (-z), lambda z: restricted_hypgeom(b, z), z
    #)


# hypgeom = hypgeom_scipy
hypgeom = hypgeom_jax


# Static

def get_th_s(gamma_s):
    return 5 / (11 - 2 * gamma_s)



def _Phi_to_c_indef_s(f, params: StaticDress):
    x = f / get_f_eq(params.gamma_s, params.c_f)
    th = get_th_s(params.gamma_s)
    return get_a_v(params.M_chirp) / f ** (5 / 3) * hypgeom(th, -(x ** (-5 / (3 * th))))



def _t_to_c_indef_s(f, params: StaticDress):
    th = get_th_s(params.gamma_s)
    return (
        5
        * get_a_v(params.M_chirp)
        / (16 * pi * f ** (8 / 3))
        * hypgeom(th, -params.c_f * f ** ((2 * params.gamma_s - 11) / 3))
    )



def d2Phi_dt2_s(f, params: StaticDress):
    return (
        12
        * pi ** 2
        * (f ** (11 / 3) + params.c_f * f ** (2 * params.gamma_s / 3))
        / (5 * get_a_v(params.M_chirp))
    )



def make_static_dress(
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> StaticDress:
    rho_s = get_rho_s(rho_6, m_1, gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return StaticDress(gamma_s, c_f, M_chirp, Phi_c, tT_c, dL, f_c)


# Dynamic

def get_f_b(m_1, m_2, gamma_s):
    """
    Gets the break frequency for a dynamic dress. This scaling relation was
    derived from fitting `HaloFeedback` runs.
    """
    beta = 0.8162599280541165
    alpha_1 = 1.441237217113085
    alpha_2 = 0.4511442198433961
    rho = -0.49709119294335674
    gamma_r = 1.4395688575650551

    return (
        beta
        * (m_1 / (1e3 * MSUN)) ** (-alpha_1)
        * (m_2 / MSUN) ** alpha_2
        * (1 + rho * np.log(gamma_s / gamma_r))
    )



def get_f_b_d(params: DynamicDress):
    """
    Gets the break frequency for a dynamic dress using our scaling relation
    derived from fitting `HaloFeedback` runs.
    """
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    return get_f_b(m_1, m_2, params.gamma_s)


def get_th_d():
    GAMMA_E = 5 / 2
    return 5 / (2 * GAMMA_E)


def get_lam(gamma_s):
    GAMMA_E = 5 / 2
    return (11 - 2 * (gamma_s + GAMMA_E)) / 3


def get_eta(params: DynamicDress):
    GAMMA_E = 5 / 2
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    rho_s = get_rho_s(params.rho_6, m_1, params.gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, params.gamma_s)
    f_eq = get_f_eq(params.gamma_s, c_f)
    f_t = get_f_b_d(params)
    return (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - params.gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * params.gamma_s) / 3)
    )


def _Phi_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    th = get_th_d()
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    v1 = ((G/C**3) * np.pi * (m_1 + m_2) * f)**(1/3)
    return (
        get_a_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - get_eta(params)
            * x ** (-get_lam(params.gamma_s))
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )# + get_a_v(params.M_chirp) / f ** (5 / 3)# * (params.PN * phasePN(nu1, v1))


def _t_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    coeff = (
        get_a_v(params.M_chirp)
        * x ** (-lam)
        / (16 * pi * (1 + lam) * (8 + 3 * lam) * f ** (8 / 3))
    )
    term_1 = 5 * (1 + lam) * (8 + 3 * lam) * x ** lam
    term_2 = 8 * lam * (8 + 3 * lam) * eta * hypgeom(th, -(x ** (-5 / (3 * th))))
    term_3 = (
        -40
        * (1 + lam)
        * eta
        * hypgeom(
            -1 / 5 * th * (8 + 3 * lam),
            -(x ** (5 / (3 * th))),
        )
    )
    term_4 = (
        -8
        * lam
        * eta
        * (
            3
            + 3 * lam
            + 5
            * hypgeom(
                1 / 5 * th * (8 + 3 * lam),
                -(x ** (-5 / (3 * th))),
            )
        )
    )
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    v1 = ((G/C**3) * pi * (m_1 + m_2) * f)**(1/3)
    return coeff * (term_1 + term_2 + term_3 + term_4)# + 5 * get_a_v(params.M_chirp) / (16 * pi * f ** (8 / 3))# *(params.PN * tPN(nu1, v1))


def d2Phi_dt2_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    nu1 = m_1 * m_2/(m_1 + m_2)**2
    newton = 12 * pi ** 2 * f ** (11 / 3) / (5 * get_a_v(params.M_chirp))
    return (
        12
        * pi ** 2
        * f ** (11 / 3)
        * x ** lam
        * (1 + x ** (5 / (3 * th)))
        / (
            get_a_v(params.M_chirp)
            * (
                5 * x ** lam
                - 5 * eta
                - 3 * eta * lam
                + x ** (5 / (3 * th)) * (5 * x ** lam - 3 * eta * lam)
                + 3
                * (1 + x ** (5 / (3 * th)))
                * eta
                * lam
                * hypgeom(th, -(x ** (-5 / (3 * th))))
            )
        )
    )#+ params.PN * (-ddphasePN(f, m_1, m_2, nu1) - newton)


def make_dynamic_dress(
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> DynamicDress:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return DynamicDress(gamma_s, rho_6, M_chirp, m_2 / m_1, Phi_c, tT_c, dL, f_c)

# Dynamic Shifted - shift phase by a constant multiplicative factor to simulate an unknown systematic

def get_f_b(m_1, m_2, gamma_s):
    """
    Gets the break frequency for a dynamic dress. This scaling relation was
    derived from fitting `HaloFeedback` runs.
    """
    beta = 0.8162599280541165
    alpha_1 = 1.441237217113085
    alpha_2 = 0.4511442198433961
    rho = -0.49709119294335674
    gamma_r = 1.4395688575650551

    return (
        beta
        * (m_1 / (1e3 * MSUN)) ** (-alpha_1)
        * (m_2 / MSUN) ** alpha_2
        * (1 + rho * np.log(gamma_s / gamma_r))
    )


def get_f_b_d(params: DynamicDressShift):
    """
    Gets the break frequency for a dynamic dress using our scaling relation
    derived from fitting `HaloFeedback` runs.
    """
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    return get_f_b(m_1, m_2, params.gamma_s)


def get_th_d():
    GAMMA_E = 5 / 2
    return 5 / (2 * GAMMA_E)


def get_lam(gamma_s):
    GAMMA_E = 5 / 2
    return (11 - 2 * (gamma_s + GAMMA_E)) / 3


def get_eta(params: DynamicDressShift):
    GAMMA_E = 5 / 2
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    rho_s = get_rho_s(params.rho_6, m_1, params.gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, params.gamma_s)
    f_eq = get_f_eq(params.gamma_s, c_f)
    f_t = get_f_b_d(params)
    return (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - params.gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * params.gamma_s) / 3)
    )


def _Phi_to_c_indef_ds(f, params: DynamicDressShift):
    f_t = get_f_b_d(params)
    x = f / f_t
    th = get_th_d()
    return params.ig * (
        get_a_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - get_eta(params)
            * x ** (-get_lam(params.gamma_s))
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )


def _t_to_c_indef_ds(f, params: DynamicDressShift):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    coeff = (
        get_a_v(params.M_chirp)
        * x ** (-lam)
        / (16 * pi * (1 + lam) * (8 + 3 * lam) * f ** (8 / 3))
    )
    term_1 = 5 * (1 + lam) * (8 + 3 * lam) * x ** lam
    term_2 = 8 * lam * (8 + 3 * lam) * eta * hypgeom(th, -(x ** (-5 / (3 * th))))
    term_3 = (
        -40
        * (1 + lam)
        * eta
        * hypgeom(
            -1 / 5 * th * (8 + 3 * lam),
            -(x ** (5 / (3 * th))),
        )
    )
    term_4 = (
        -8
        * lam
        * eta
        * (
            3
            + 3 * lam
            + 5
            * hypgeom(
                1 / 5 * th * (8 + 3 * lam),
                -(x ** (-5 / (3 * th))),
            )
        )
    )
    return coeff * (term_1 + term_2 + term_3 + term_4)


def d2Phi_dt2_ds(f, params: DynamicDressShift):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    return params.ig * (
        12
        * pi ** 2
        * f ** (11 / 3)
        * x ** lam
        * (1 + x ** (5 / (3 * th)))
        / (
            get_a_v(params.M_chirp)
            * (
                5 * x ** lam
                - 5 * eta
                - 3 * eta * lam
                + x ** (5 / (3 * th)) * (5 * x ** lam - 3 * eta * lam)
                + 3
                * (1 + x ** (5 / (3 * th)))
                * eta
                * lam
                * hypgeom(th, -(x ** (-5 / (3 * th))))
            )
        )
    )


def make_dynamic_dress_shift(ig,
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> DynamicDressShift:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return DynamicDressShift(ig, gamma_s, rho_6, M_chirp, m_2 / m_1, Phi_c, tT_c, dL, f_c)

# Accretion Disk

def _Phi_to_c_indef_a(f, params: AccretionDisk):#2pi intdf f dt/df between fc and f
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)

    totm = M1 + M2

    #inout = -1 is inwards torque, so speeds up inspiral

    # 'z' position i.e. last argument of hyp2f1

    zf = params.inout * 64 * np.sqrt(2/3) * f**(8/3) * G * M1**3\
     * np.pi**(8/3)/(5 * C**4 * (G * totm**4)**(1/3) * params.SigM2)

    def complog(z):

        return np.log(np.absolute(z))+1j*np.angle(z)

    # hyp2f1 expansion in terms of complex logs for hyp2f1(3/8,1,11/8,z)
    def hypgeomacc(z1):
        return (-3*complog(1 - z1**0.125))/(8.*z1**0.375)\
            -(0.375*1j*complog(1 - 1j*z1**0.125))/z1**0.375\
            +(0.375*1j*complog(1 + 1j*z1**0.125))/z1**0.375\
            +(3*complog(1 + z1**0.125))/(8.*z1**0.375)\
            -(3*(-1)**0.75*complog(1 - (z1**0.125)/np.exp((1j*np.pi)/4.)))/(8.*z1**0.375)\
            +(3*(-1)**0.25*complog(1 - np.exp((1j*np.pi)/4.)*z1**0.125))/(8.*z1**0.375)\
            -(3*(-1)**0.25*complog(1 - (z1**0.125)/np.exp((3*1j*np.pi)/4.)))/(8.*z1**0.375)\
            +(3*(-1)**0.75*complog(1 - np.exp((3*1j*np.pi)/4.)*z1**0.125))/(8.*z1**0.375)


    return (
        params.inout * 4 * np.sqrt(2/3) * C * f * M1**2 * totm * np.pi  * hyp2f1(3/8,1,11/8,zf)\
            /(3 * M2 * (G * totm)**(2/3) * (G * totm**4)**(
 1/3) * params.SigM2)
    )


def _t_to_c_indef_a(f, params: AccretionDisk):# -int dfdt/df betweem fc and f
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)
    totm=M1 + M2
    return (-
        (C * M1**2 * (totm/G**2)**(
 1/3) * (-params.inout * 8 * np.log(f) + params.inout *  
   3 *np.log(
     128 *f**(8/3) * G * M1**3 * np.pi**(8/3) - params.inout *
      5 * np.sqrt(6) * C**4 * (G * totm**4)**(1/3) * params.SigM2))/(6 * np.sqrt(6) * M2 * (G * totm**4)**(
 1/3) * params.SigM2)
    )
    )


def d2Phi_dt2_a(f, params: AccretionDisk):
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)
    totm = M1 + M2
    return (-((3 * f**3 * M2 * ((G * totm)/f**2)**(2/3) * np.pi\
         * (-params.inout * 128 * f**2 * G * M1**3 * np.pi**(8/3) +5\
          * np.sqrt(6) * C**3 * totm * np.sqrt(G *totm)* ((G *totm)/f**2)**(1/6)\
           * np.sqrt(C**2/(f**2 *((G *totm)/f**2)**(2/3))) * params.SigM2)/\
           (10 * C**5 * M1**2 * totm)))
    )


def make_accretion_disk(
    m_1,
    m_2,
    SigM2,
    inout,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> AccretionDisk:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return AccretionDisk(SigM2, inout, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)

# Accretion Disk Shifted


def _Phi_to_c_indef_as(f, params: AccretionDiskShift):#2pi intdf f dt/df between fc and f
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)

    totm = M1 + M2

    #inout = -1 is inwards torque, so speeds up inspiral

    # 'z' position i.e. last argument of hyp2f1

    zf = params.inout * 64 * np.sqrt(2/3) * f**(8/3) * G * M1**3\
     * np.pi**(8/3)/(5 * C**4 * (G * totm**4)**(1/3) * params.SigM2)

    def complog(z):

        return np.log(np.absolute(z))+1j*np.angle(z)

    # hyp2f1 expansion in terms of complex logs for hyp2f1(3/8,1,11/8,z)
    def hypgeomacc(z1):
        return (-3*complog(1 - z1**0.125))/(8.*z1**0.375)\
            -(0.375*1j*complog(1 - 1j*z1**0.125))/z1**0.375\
            +(0.375*1j*complog(1 + 1j*z1**0.125))/z1**0.375\
            +(3*complog(1 + z1**0.125))/(8.*z1**0.375)\
            -(3*(-1)**0.75*complog(1 - (z1**0.125)/np.exp((1j*np.pi)/4.)))/(8.*z1**0.375)\
            +(3*(-1)**0.25*complog(1 - np.exp((1j*np.pi)/4.)*z1**0.125))/(8.*z1**0.375)\
            -(3*(-1)**0.25*complog(1 - (z1**0.125)/np.exp((3*1j*np.pi)/4.)))/(8.*z1**0.375)\
            +(3*(-1)**0.75*complog(1 - np.exp((3*1j*np.pi)/4.)*z1**0.125))/(8.*z1**0.375)


    return params.ig * (
        params.inout * 4 * np.sqrt(2/3) * C * f * M1**2 * totm * np.pi  * hyp2f1(3/8,1,11/8,zf)\
            /(3 * M2 * (G * totm)**(2/3) * (G * totm**4)**(
 1/3) * params.SigM2)
    )


def _t_to_c_indef_as(f, params: AccretionDiskShift):# -int dfdt/df betweem fc and f
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)
    totm=M1 + M2
    return (-
        (C * M1**2 * (totm/G**2)**(
 1/3) * (-params.inout * 8 * np.log(f) + params.inout *  
   3 * np.log(
     128 *f**(8/3) * G * M1**3 * np.pi**(8/3) - params.inout *
      5 * np.sqrt(6) * C**4 * (G * totm**4)**(1/3) * params.SigM2))/(6 * np.sqrt(6) * M2 * (G * totm**4)**(
 1/3) * params.SigM2)
    )
    )


def d2Phi_dt2_as(f, params: AccretionDiskShift):
    M1 = get_m_1(params.M_chirp, params.q)
    M2 = get_m_2(params.M_chirp, params.q)
    totm = M1 + M2
    return params.ig * (-((3 * f**3 * M2 * ((G * totm)/f**2)**(2/3) * np.pi\
         * (-params.inout * 128 * f**2 * G * M1**3 * np.pi**(8/3) +5\
          * np.sqrt(6) * C**3 * totm * np.sqrt(G *totm)* ((G *totm)/f**2)**(1/6)\
           * np.sqrt(C**2/(f**2 *((G *totm)/f**2)**(2/3))) * params.SigM2)/\
           (10 * C**5 * M1**2 * totm)))
    )

#@jit
def make_accretion_disk_shift(IG,
    m_1,
    m_2,
    SigM2,
    inout,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> AccretionDisk:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return AccretionDiskShift(IG, SigM2, inout, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)

# GravAtom

def master_call(f, params: GravAtom):
    ga_results = gatom_interp(get_m_1(params.M_chirp, params.q)/MSUN,
    params.alpha, params.q, params.epsilon_init,
    R_211, ion_r_co_211, ion_E_co_211)

    return ga_results[7], ga_results[6], ga_results[8]


#@jit
def make_grav_atom(
    m_1,
    m_2,
    alpha,
    epsilon_init,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> GravAtom:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return GravAtom(alpha, epsilon_init, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)

# GravAtom Shifted

def master_call_shift(f, params: GravAtomShift):
    ga_results = gatom_interp(get_m_1(params.M_chirp, params.q)/MSUN,
    params.alpha, params.q, params.epsilon_init,
    R_211, ion_r_co_211, ion_E_co_211)

    return ga_results[7], ga_results[6], ga_results[8]


def make_grav_atom_shift(IG,
    m_1,
    m_2,
    alpha,
    epsilon_init,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> GravAtom:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return GravAtomShift(IG, alpha, epsilon_init, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)

# Surrogate dynamic dark dress
def _Phi_to_c_indef_sur(f, params: SurrogateDD):
    input_ = np.array([get_m_1(params.M_chirp,params.q)/MSUN, get_m_2(params.M_chirp,params.q)/MSUN, params.rho_6 / (1e16 * MSUN/(PC**3)), params.gamma_s])
    phi_interp = interp1d(getPrediction(model,input_)[0][::-1],np.log(np.array(getPrediction(model,input_)[1][::-1],dtype=float)), fill_value = 'extrapolate', kind = 'cubic')
    return np.exp(phi_interp(f))

def _t_to_c_indef_sur(f, params: SurrogateDD):
    input_ = np.array([get_m_1(params.M_chirp,params.q)/MSUN, get_m_2(params.M_chirp,params.q)/MSUN, params.rho_6 / (1e16 * MSUN/(PC**3)), params.gamma_s])
    t_interp = interp1d(getPrediction(model,input_)[0][::-1],getPrediction(model,input_)[2][::-1], fill_value = 'extrapolate', kind = 'cubic')
    return t_interp(f)

def d2Phi_dt2_sur(f, params: SurrogateDD):
    input_ = np.array([get_m_1(params.M_chirp,params.q)/MSUN, get_m_2(params.M_chirp,params.q)/MSUN, params.rho_6 / (1e16 * MSUN/(PC**3)), params.gamma_s])
    phidd_interp = interp1d(getPrediction(model,input_)[0][::-1],np.log(np.array(getPrediction(model,input_)[3][::-1],dtype=float)), fill_value = 'extrapolate', kind = 'cubic')
    return np.exp(phidd_interp(f))

def make_surr_dd(
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> GravAtom:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return SurrogateDD(gamma_s, rho_6, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)


### Combo - not working properly yet 


def _Phi_to_c_indef_c(f, params: Combo):#2pi intdf f dt/df between fc and f
    v_count = 0
    dm_phase = 0
    ad_phase = 0
    ga_phase = 0
    if params.rho_6 != 0:
        dm_phase = _Phi_to_c_indef_d(f, params)
        v_count += 1
    if params.SigM2 != 0:
        ad_phase = _Phi_to_c_indef_a(f, params)
        v_count += 1
    if params.epsilon_init !=0:
        ga_phase = master_call(f, params)[0](f)
        v_count += 1
    # if no environments, just returns vacuum
    return (1 - v_count) * _Phi_to_c_indef_v(f, params) + dm_phase + ad_phase + ga_phase


def _t_to_c_indef_c(f, params: Combo):# -int dfdt/df betweem fc and f

    return -2 * _t_to_c_indef_v(f, params) + _t_to_c_indef_d(f, params) + _t_to_c_indef_a(f, params) + master_call(f, params)[1](f)
    


def d2Phi_dt2_c(f, params: Combo):

    return -2 * d2Phi_dt2_v(f, params) + d2Phi_dt2_d(f, params) + d2Phi_dt2_a(f, params) + master_call(f, params)[2](f)

def make_combo(
    m_1,
    m_2,
    gamma_s,
    rho_6,
    SigM2,
    inout,
    alpha,
    epsilon_init,
    Phi_c=np.array(0.0),
    t_c=None,
    dL=np.array(1e8 * PC),
) -> Combo:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = np.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1)
    return Combo(gamma_s, rho_6, SigM2, inout, alpha, epsilon_init, M_chirp, m_2/m_1, Phi_c, tT_c, dL, f_c)


def get_f_range(params: Binary, t_obs: float, bracket=None) -> Tuple[float, float]:
    """
    Finds the frequency range [f(-(t_obs + tT_c)), f(-tT_c)].
    """
    # Find frequency t_obs + tT_c before merger
    if bracket is None:
        bracket = (params.f_c * 0.001, params.f_c * 1.1)

#    fn = lambda f_l: (jax.jit(t_to_c)(f_l, params) - (t_obs + params.tT_c)) ** 2
    fn = lambda f_l: (t_to_c(f_l, params) - (t_obs + params.tT_c)) ** 2
    res = minimize_scalar(fn, bracket = bracket)#bounds=bracket,
    if not res.success:
        raise RuntimeError(f"finding f_l failed: {res}")
    f_l = res.x

    # Find frequency tT_c before merger
#    fn = lambda f_h: (jax.jit(t_to_c)(f_h, params) - params.tT_c) ** 2
    fn = lambda f_h: (t_to_c(f_h, params) - params.tT_c) ** 2
    res = minimize_scalar(fn, bracket=bracket)
    if not res.success:
        raise RuntimeError(f"finding f_h failed: {res}")
    f_h = res.x

    return (f_l, f_h)


def convert(params: Binary, NewType) -> Binary:
    """
    Change binary's type by dropping attributes.
    """
    if isinstance(params, DynamicDress) and NewType is StaticDress:
        m_1 = get_m_1(params.M_chirp, params.q)
        m_2 = get_m_2(params.M_chirp, params.q)
        rho_s = get_rho_s(params.rho_6, m_1, params.gamma_s)
        c_f = get_c_f(m_1, m_2, rho_s, params.gamma_s)
        return StaticDress(
            params.gamma_s,
            c_f,
            params.M_chirp,
            params.Phi_c,
            params.tT_c,
            params.dL,
            params.f_c,
        )
    elif (
        isinstance(params, StaticDress) or isinstance(params, DynamicDress) or isinstance(params, AccretionDisk) or isinstance(params, GravAtom)
    ) and NewType is VacuumBinary:
        return VacuumBinary(**{f: getattr(params, f) for f in VacuumBinary._fields})
    else:
        raise ValueError("invalid conversion")
