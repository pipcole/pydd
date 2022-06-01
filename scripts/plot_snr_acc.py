import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm.auto import tqdm

import jax.numpy as jnp

from pydd.analysis import calculate_SNR
from pydd.binary import *
from utils import DL_BM

"""
Plots SNRs for GR-in-vacuum binaries as a function of chirp mass and luminosity
distance.

Produces `figures/snrs.pdf`.
"""

if __name__ == "__main__":
    t_obs_lisa = 5 * YR # if you change this may need to change brackets for finding f_low below
      #was 1e2 # get_f_isco(1e3 * MSUN)
    M_chirp_min = 10 * MSUN # get_M_chirp(1e3 * MSUN, 1 * MSUN)
    M_chirp_max = 2000 * MSUN # get_M_chirp(1e4 * MSUN, 10 * MSUN)
    dL_min = 1e6 * PC
    dL_max = 1e10 * PC
    print(get_M_chirp(1e3 * MSUN, 1 * MSUN))

    M_chirps = jnp.geomspace(M_chirp_min, M_chirp_max, 40)
    dLs = jnp.geomspace(dL_min, dL_max, 35)

    snrs = np.zeros([len(M_chirps), len(dLs)])
    f_ls = np.zeros([len(M_chirps), len(dLs)])


    for i, M_chirp in enumerate(tqdm(M_chirps)):
        for j, dL in enumerate(dLs):
            f_c = get_f_isco(get_m_1(M_chirp,1e-4))
            dd_a = AccretionDisk(1e7, 30.0, M_chirp,1e-4, 0.0, 0.0, dL, f_c)

            # need to adjust brackets according to mass to find f_low for accretion disk
            if M_chirp < 8e32:
                bracket_adap = (3e-3,2e-2)
            else: bracket_adap = (1e-3,1e-2)
            if M_chirp < 8e31:
                bracket_adap = (1e-2,1e-1)

            f_l = root_scalar(
                lambda f: t_to_c(f, dd_a) - t_obs_lisa,
                bracket=bracket_adap,
                rtol=1e-15,
                xtol=1e-100,
            ).root
            f_ls[i, j] = f_l
            fs = jnp.linspace(f_l, f_c, 3000)
            snrs[i, j] = calculate_SNR(dd_a, fs)






    plt.figure(figsize=(4, 3.5))

    plt.axvline(get_M_chirp(1e3, 1.4), color="r", linestyle="--")
    plt.axhline(76, color="r", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    cs = plt.contour(
        M_chirps / MSUN,
        dLs / (1e6 * PC),
        jnp.log10(snrs.T),
        levels=jnp.linspace(-2, 6, 9).round(),
        alpha=0.8,
    )
    plt.clabel(cs, inline=True, fontsize=10, fmt=r"$10^{%i}$")

    cs = plt.contour(
        M_chirps / MSUN,
        dLs / (1e6 * PC),
        snrs.T,
        levels=[15],
        colors=["r"],
    )
    plt.clabel(cs, inline=True, fontsize=10, fmt=r"%g")

    plt.xlabel(r"$\mathcal{M}$ [M$_\odot$]")
    plt.ylabel(r"$d_L$ [Mpc]")
    plt.tight_layout()

    plt.savefig("figures/snrs_acc_flow_fc_isco.pdf")
    plt.close()


#IGNORE BELOW
    f_c_list=np.logspace(-2,2,100)
    snrs_time = []
    f_l_time=[]
    for p in range(len(f_c_list)):

        dd_a = AccretionDisk(1e7, 30.0, get_M_chirp(1e3 * MSUN, 1.4 * MSUN), 1.4/1e3, 0.0, 0.0, DL_BM, f_c_list[p])

        # need to adjust brackets according to mass/f_c to find f_low for accretion disk
        if f_c_list[p] < 1e-1:
            bracket_adap = (9e-3,3e-2)
        else: bracket_adap = (1e-2,1e-1)

        f_l = root_scalar(
            lambda f: t_to_c(f, dd_a) - t_obs_lisa,
            bracket=bracket_adap,
            rtol=1e-15,
            xtol=1e-100,
        ).root
        print(f_l)
        fs = jnp.linspace(f_l, f_c_list[i], 3000)
        snrs_time.append(calculate_SNR(dd_a, fs))
        f_l_time.append(f_l)
        plt.semilogx(fs, t_to_c(fs, dd_a)/(60*60*24*365.25))
    plt.xlabel('f [Hz]')
    plt.ylabel('$t$ [yrs]')
    plt.ylim(0,)
    plt.savefig('figures/tc_vs_f.pdf')
    plt.close()
    plt.semilogx(f_c_list,snrs_time)
    plt.xlabel('$f_c$ [Hz]')

    plt.ylabel('SNR')
    plt.savefig("figures/snr_vs_fc.pdf")
    plt.close()

    plt.semilogx(f_c_list,f_l_time)
    plt.xlabel('$f_c$ [Hz]')
    plt.ylabel('$f_l$ [Hz]')
    plt.savefig("figures/fc_vs_fl.pdf")
    plt.close()
