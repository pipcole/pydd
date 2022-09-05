import scipy.special as sc
from scipy import integrate
from math import factorial
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
#import jax.numpy as jnp
import numpy as np
from numpy import append, gradient, pi, array, linspace, sqrt, exp, loadtxt

# tryout_211 = np.loadtxt('211_1.txt', unpack=True)
#
# np.savez_compressed('try211.npz', array1=tryout_211[0], array2=tryout_211[1], array3=tryout_211[2], array4=tryout_211[3], array5=tryout_211[4], array6=tryout_211[5])
#b_211 = jnp.loadtxt('/Users/pippacole/Documents/dec-pydd-main-2/src/pydd/211_1.txt')

#b_211 = jnp.load('try211.npz')

def OmegaKepler(rs, R_star): #Kepler's third law, for the effective two-body problem insert rs*(1+q)
    return sqrt(rs/(2*R_star**3))

def PhiGR(rs, q, r0): #Phase GR-in-Vacuum
    return (1+q)**(-0.5) * (2*r0/rs)**2.5 / (32*q) - (1+q)**(-0.5) * (2*3)**2.5 / (32*q)

def Rbound(r, rs, alpha): #Wavefunction of the bound states Eq. (2.5)
    rbohr = rs / (2*alpha**2)
    return sqrt(4/(rbohr**3*96)) * exp(-r/(2*rbohr)) * (2*r/(2*rbohr))# * sc.genlaguerre(n-l-1,2*l+1)(2*r/(n*rbohr))

def dqOverdt(q, rs, r, epsilon, alpha): #Mass accretion secondary object Eq.(5.1)
    return 4*pi*q**2*rs**2 * epsilon * Rbound(r,rs,alpha)**2 * 0.11936620731892152#abs(sc.sph_harm(m,l,0,pi/2))**2

def depsilonOverdt(dqOverdt, epsilon, q, alpha, ion_rate): #Epsilon is M_cloud/M_BH, so it suffers from accretion and ionization
    return - dqOverdt - epsilon * ion_rate

def dtOverdr(q, dqOverdt, rs, r, Omega0, epsilon, alpha, ion_energy): #Eq.(5.6)
    dEGWdt = (16/5)*(q/(1+q))**2*rs*r**4*Omega0**6
    dEaccretiondt = sqrt((1+q)*rs/(2*r**3)) * ( ((q+2)/(2*(1+q)**1.5)) * sqrt(rs*r/2) - rs/(2*alpha) ) * dqOverdt
    dEionizdt = ion_energy
    return q*rs/(4*r**2) / ( - dEGWdt - dEaccretiondt - dEionizdt)

def dtOverdr_extra(q, dqOverdt, rs, r, Omega0, epsilon, alpha, ion_energy): #Eq.(5.6)
    dEGWdt = (16/5)*(q/(1+q))**2*rs*r**4*Omega0**6
    dEaccretiondt = sqrt((1+q)*rs/(2*r**3)) * ( ((q+2)/(2*(1+q)**1.5)) * sqrt(rs*r/2) - rs/(2*alpha) ) * dqOverdt
    dEionizdt = ion_energy
    return q*rs/(4*r**2) / ( - dEGWdt - dEaccretiondt - dEionizdt)


#F#For a |211> cloud, we import the following tabulated data, can be rescaled for arbritary alpha, q, M_cloud
# R_211, ion_r_co_211, ion_E_co_211, PionOverPGW_co_211, ion_r_count_211, ion_E_count_211, PionOverPGW_count_211 = jnp.loadtxt('211.txt', ujnpack=True)
R_211, ion_r_co_211, ion_E_co_211, PionOverPGW_co_211, ion_r_count_211, ion_E_count_211, PionOverPGW_count_211 = loadtxt('/Users/pippacole/Documents/dec-pydd-main-2/src/pydd/211_1.txt',unpack=True)#b_211['array1'], b_211['array2'], b_211['array3'], b_211['array4'], b_211['array5'], b_211['array6'], b_211['array7']
# R_322, ion_r_co_322, ion_E_co_322, PionOverPGW_co_322, ion_r_count_322, ion_E_count_322, PionOverPGW_count_322 = jnp.loadtxt('322.txt', ujnpack=True)
# R_311, ion_r_co_311, ion_E_co_311, PionOverPGW_co_311, ion_r_count_311, ion_E_count_311, PionOverPGW_count_311 = jnp.loadtxt('311.txt', ujnpack=True)
# R_320, ion_r_co_320, ion_E_co_320, PionOverPGW_co_320, ion_r_count_320, ion_E_count_320, PionOverPGW_count_320 = jnp.loadtxt('320.txt', ujnpack=True)
# R_210, ion_r_co_210, ion_E_co_210, PionOverPGW_co_210, ion_r_count_210, ion_E_count_210, PionOverPGW_count_210 = jnp.loadtxt('210.txt', ujnpack=True)

R_211 /= 2 #Divide by 2 to work in units of rs = 1
# R_322 /= 2
# R_311 /= 2
# R_320 /= 2
# R_210 /= 2

#Some definitions

#n, l, m are the principal quantum numbers, so in this case 2, 1, 1
#We work in the limit of small q, so q <= 1e-3
#We are only dealing with equatorial orbits, chi = 0 means co-rotating, chi = jnp.pi means counter-rotating
def gatom_interp(MBH, alpha, q_init, epsilon_init, R, rate_co, energy_co):

    solar_rs = 2953.36084797935 # 2*G*Msun/c**2
    c = 299792458 * (365.25*24*3600) # Includes the conversion from seconds to years
    rs = 1

    rbohr = rs/(2*alpha**2)
    r_init = 256**(1/3) * rbohr # 2 * 2*2**2*rbohr #now set up for n=2 # 256**(1/3) * rbohr # this starts exactly on "2nd-last" discontinuity. previous earlier time 2 * 2*n**2*rbohr
    t_init = 0.
    number_of_r_steps = 1000 #1000 #300
    r = linspace(r_init, 3*rs, number_of_r_steps)
    j = len(R)-2

    t = array([t_init])
    q = array([q_init])
    epsilon = array([epsilon_init])
    Phi = array([0])

    #Hand-written version of Runge-Kutta order 3
    for i in range(number_of_r_steps-1):

        dr = r[i+1]-r[i]

        Omega0 = OmegaKepler((1+q[i])*rs, r[i])
        while (R[j]*(0.2/alpha)**2 > r[i]):
            j -= 1
#        if chi == 0: #We take the weighted average between two values (either rate or energy) at some orbital separation closest to the actual separation
        ion_rate = (q[i]/0.001)**2 * (alpha/0.2)**3 * (rate_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]) + rate_co[j]*(-R[j]*(0.2/alpha)**2+r[i])) / ((R[j+1]-R[j])*(0.2/alpha)**2)
        ion_energy = (epsilon[i]/0.01) * (q[i]/0.001)**2 * (alpha/0.2)**5 * (energy_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]) + energy_co[j]*(-R[j]*(0.2/alpha)**2+r[i])) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#        else:
#            ion_rate = (q[i]/0.001)**2 * (alpha/0.2)**3 * (rate_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]) + rate_counter[j]*(-R[j]*(0.2/alpha)**2+r[i])) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#            ion_energy = (epsilon[i]/0.01) * (q[i]/0.001)**2 * (alpha/0.2)**5 * (energy_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]) + energy_counter[j]*(-R[j]*(0.2/alpha)**2+r[i])) / ((R[j+1]-R[j])*(0.2/alpha)**2)

        k1q = dqOverdt(q[i],rs,r[i],epsilon[i],alpha)
        k1t = dr * dtOverdr(q[i],k1q,rs,r[i],Omega0,epsilon[i],alpha,ion_energy)
        k1epsilon = k1t * depsilonOverdt(k1q,epsilon[i],q[i],alpha,ion_rate)
        k1q *= k1t
        k1Phi = Omega0 * k1t

        Omega0 = OmegaKepler((1+q[i]+k1q/2)*rs,r[i]+dr/2)
        while (R[j]*(0.2/alpha)**2 > r[i]+dr/2):
            j -= 1
#        if chi == 0:
        ion_rate = ((q[i]+k1q/2)/0.001)**2 * (alpha/0.2)**3 * (rate_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr/2) + rate_co[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr/2)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
        ion_energy = ((epsilon[i]+k1epsilon/2)/0.01) * ((q[i]+k1q/2)/0.001)**2 * (alpha/0.2)**5 * (energy_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr/2) + energy_co[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr/2)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#        else:
#            ion_rate = ((q[i]+k1q/2)/0.001)**2 * (alpha/0.2)**3 * (rate_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr/2) + rate_counter[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr/2)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#            ion_energy = ((epsilon[i]+k1epsilon/2)/0.01) * ((q[i]+k1q/2)/0.001)**2 * (alpha/0.2)**5 * (energy_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr/2) + energy_counter[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr/2)) / ((R[j+1]-R[j])*(0.2/alpha)**2)

        k2q = dqOverdt(q[i]+k1q/2,rs,r[i]+dr/2,epsilon[i]+k1epsilon/2,alpha)
        k2t = dr * dtOverdr(q[i]+k1q/2,k2q,rs,r[i]+dr/2,Omega0,epsilon[i]+k1epsilon/2,alpha,ion_energy)
        k2epsilon = k2t * depsilonOverdt(k2q,epsilon[i]+k1epsilon/2,q[i]+k1q/2,alpha,ion_rate)
        k2q *= k2t
        k2Phi = Omega0 * k2t

        Omega0 = OmegaKepler((1+q[i]-k1q+2*k2q)*rs,r[i]+dr)
        while (R[j]*(0.2/alpha)**2 > r[i]+dr):
            j -= 1
#        if chi == 0:
        ion_rate = ((q[i]-k1q+2*k2q)/0.001)**2 * (alpha/0.2)**3 * (rate_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr) + rate_co[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
        ion_energy = ((epsilon[i]-k1epsilon+2*k2epsilon)/0.01) * ((q[i]-k1q+2*k2q)/0.001)**2 * (alpha/0.2)**5 * (energy_co[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr) + energy_co[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#        else:
#            ion_rate = ((q[i]-k1q+2*k2q)/0.001)**2 * (alpha/0.2)**3 * (rate_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr) + rate_counter[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr)) / ((R[j+1]-R[j])*(0.2/alpha)**2)
#            ion_energy = ((epsilon[i]-k1epsilon+2*k2epsilon)/0.01) * ((q[i]-k1q+2*k2q)/0.001)**2 * (alpha/0.2)**5 * (energy_counter[j+1]*(R[j+1]*(0.2/alpha)**2-r[i]-dr) + energy_counter[j]*(-R[j]*(0.2/alpha)**2+r[i]+dr)) / ((R[j+1]-R[j])*(0.2/alpha)**2)

        k3q = dqOverdt(q[i]-k1q+2*k2q,rs,r[i]+dr,epsilon[i]-k1epsilon+2*k2epsilon,alpha)
        k3t = dr * dtOverdr(q[i]-k1q+2*k2q,k3q,rs,r[i]+dr,Omega0,epsilon[i]-k1epsilon+2*k2epsilon,alpha,ion_energy)
        k3epsilon = k3t * depsilonOverdt(k3q,epsilon[i]-k1epsilon+2*k2epsilon,q[i]-k1q+2*k2q,alpha,ion_rate)
        k3q *= k3t
        k3Phi = Omega0 * k3t

        dt = k1t/6 + 4*k2t/6 + k3t/6
        dq = k1q/6 + 4*k2q/6 + k3q/6
        depsilon = k1epsilon/6 + 4*k2epsilon/6 + k3epsilon/6
        dPhi = 2 * (k1Phi/6 + 4*k2Phi/6 + k3Phi/6)

        t = append(t, t[i] + dt)
        q = append(q, q[i] + dq)
        epsilon = append(epsilon, epsilon[i] + depsilon)
        Phi = append(Phi, Phi[i] + dPhi)

    t *= MBH*solar_rs/c
    f = (1/pi)*OmegaKepler((1+q)*rs,r)/(MBH*solar_rs/299792458)

    t_interp = interp1d(f, -t*(60*60*24*365.25), fill_value = 'extrapolate', kind = 'cubic')
    Phi_interp = interp1d(f, -Phi, fill_value = 'extrapolate', kind = 'cubic')
    deriv = (gradient(Phi, f))**-1*f*4*pi**2
    Phi_dd_interp = interp1d(f, deriv, fill_value = 'extrapolate', kind = 'cubic')

    return t, r, f, q, epsilon, Phi, t_interp, Phi_interp, Phi_dd_interp


#Set up a function that calculates the relevant quantities
#n, l, m are the principal quantum numbers, so in this case 2, 1, 1
#We work in the limit of small q, so q equal or smaller than 1e-3
#We are only dealing with equatorial orbits, chi = 0 means co-rotating, chi = np.pi means counter-rotating
