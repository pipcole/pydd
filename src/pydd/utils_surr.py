import numpy as np

from typing import Tuple
from scipy.integrate import cumtrapz

import os
import pySurrogate
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# Physical Constants
GeVc2 = 1.78266192e-27 # kg
G = 6.67408e-11 # m3/kg/s2, The Newtonian gravitational constant [Legacy 2012: 6.67408]
c = 299792458 # m/s, The Speed of light in a vacuum

# Useful Unit Conversions
pc = 3.08567758149137e16 # m, Parsec to metres
Mo = 1.98855e30 # kg, Solar Mass to kg
yr = 365.25 * 24 * 3600 # s, A year in s
day = 60*60*24 # s, A day in s

def getRisco(m: float) -> float:
  """ Calculates the radius [pc] of the Innermost Stable Circular Orbit
  for a massive object of mass m [M_sun].
  """
  
  return 6 *G *m *Mo/c**2 /pc # [pc] # = 3 Rs

def getPeriodFromDistance(a: float, M_tot: float) -> float:
  """ Calculates the Kepplerian period [s] of binary system with
  total mass [M_sun] M_tot at a semi-major axis a [pc].
  """

  return (a *pc)**(3/2) *2*np.pi / np.sqrt(G *M_tot *Mo)

def getVacuumMergerDistance(m1: float, m2: float, t: float) -> float:
  """ Calculates the distance [pc] at which a binary in vacuum would
  merge after t [s].

  * m1, m2 are the masses [M_sun] of the components.
  * t is the time [s] until merger event.
  """

  return (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3 / (5 *c**5) *t)**(1/4) /pc

def dr2dt(r2: float, m1: float, m2: float, rho_DM_at_r2: float, separate: bool = False) -> float:
  """ Calculates the time derivative [pc/s] of the seperation between the
  two components in a binary (m1, m2) due to gravitational and dynamical
  friction (with a dark matter distribution) energy losses as defined by Equation 2.6
  of arxiv.org/abs/2002.12811.

  * r2 is the seperation [pc] between the two components.
  * m1, m2 are the masses [M_sun] of the two components.
  * rho_DM_at_r2 [M_sun/pc3] is the effective density at distance r2 of dark
  matter particles (which are faster than the orbital velocity at that distance).
  
  If separate, returns a list of two elements for the gravitational and dynamic friction part.
  """

  M = m1 +m2
  Lambda = np.sqrt(m1/m2)

  c_ = c /pc # [pc/s]
  G_ = G *Mo/pc**3 # [pc3/M_sun/s2]

  dr2dt_GW = - 64 * G_**3 *M *m1 *m2/(5 *c_**5 *r2**3)
  dr2dt_DF = - 8 *np.pi *np.sqrt(G_/M) *(m2/m1) *np.log(Lambda) *rho_DM_at_r2 *r2**(5/2)

  if separate:
    return dr2dt_GW, dr2dt_DF # [pc/s]
  else:
    return dr2dt_GW +dr2dt_DF # [pc/s]

def getDephasingFromDensity(r: np.array, rho: np.array, m1: float, m2: float) -> Tuple[np.array, np.array, np.array]:
  """ Creates the time and orbital frequency evolution of the dephasing until coalescence induced
  on the binary from its vacuum evolution, because of an effective density profile rho(r).
  
  * r is the binary separation [pc] for which the effective density is rho(r).
  * rho is the effective density [M_sun/pc3] at a given binary separation.
  * m1, m2 are the masses [M_sun] of the binary components.
  
  Returns t, fGW/2, dPhase
  """
  # Construct the quantity inside of the integral
  fGW = 2/getPeriodFromDistance(r, m1 +m2)

  drdtGW, drdtDF = dr2dt(r, m1, m2, rho, separate = True)
  integrand = fGW *-drdtDF/drdtGW/(drdtGW +drdtDF)

  dPhase = 2 *np.pi *cumtrapz(integrand, r, initial = 0)

  # Calculate the time evolution for the binary
  t = cumtrapz(1/dr2dt(np.flip(r), m1, m2, rho), np.flip(r), initial = 0)

  return t, np.flip(fGW)/2, np.flip(dPhase), # [s, Hz, rads]

def getChirpMass(m1: float, m2: float) -> float:
  """ Calculates the chirp mass in the given units."""
  return (m1 *m2)**(3/5) / (m1 +m2)**(1/5)

def getVacuumPhase(fGW: float, m1: float, m2: float) -> float:
  """ Calculates the phase [rad] of the gravitational wave in the vacuum case in
  the Newtonian approximation.

  * m1, m2 are the masses [M_sun] of the two components.
  * fGW is the frequency of the gravitational wave.
  """

  return 1/16 *(c**3 / (np.pi *G *getChirpMass(m1, m2) *Mo *fGW))**(5/3)