from math import pi
import os

import click
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
import matplotlib
from matplotlib import pyplot as plt

from pydd.analysis import calculate_SNR, calculate_match_unnormd_fft, get_match_pads
from pydd.binary import (
    DynamicDress,
    MSUN,
    Phi_to_c,
    VacuumBinary,
    convert,
    get_M_chirp,
    get_rho_s,
    AccretionDisk,
)
from utils import (
    M_1_BM,
    M_2_BM,
    get_loglikelihood_fn_v,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_system,
    setup_system_acc,
)

"""
For dark dresses with fixed BH masses and various rho_6 and gamma_s values,
computes the naive dephasing, best-fit vacuum system and its dephasing, chirp
mass bias and SNR loss.
"""


def fit_v(dd_s: AccretionDisk, f_l) -> VacuumBinary:
    """
    Find best-fit vacuum system.
    """
    loglikelihood_v = get_loglikelihood_fn_v(dd_s, f_l, dd_s.f_c, n_f=100000)
    fun = lambda x: -loglikelihood_v(x)
    bracket = (dd_s.M_chirp / MSUN -0.5, dd_s.M_chirp / MSUN +0.5)
    res = minimize_scalar(fun, bracket, tol=1e-30)#changed tol from 1e-15

    assert res.success, res


    return VacuumBinary(
        res.x * MSUN,
        dd_s.Phi_c,
        dd_s.tT_c,
        dd_s.dL,
        dd_s.f_c,
    )


def get_M_chirp_err(dd_v: VacuumBinary, dd_s: AccretionDisk, f_l) -> jnp.ndarray:
    """
    Returns an estimate of the error on the best-fit vacuum system's chirp
    mass.
    """
    M_chirp_MSUN = dd_v.M_chirp / MSUN
    ngrid = 500
    M_chirp_MSUN_grid = jnp.linspace(M_chirp_MSUN - 0.00005, M_chirp_MSUN + 0.00005, ngrid)
    # Wrap and jit
    ll = get_loglikelihood_fn_v(dd_s, f_l, dd_s.f_c, n_f=100000)
    loglikelihoods = jnp.array([ll(x) for x in M_chirp_MSUN_grid])

    lls_scaled = loglikelihoods - loglikelihoods[int(ngrid/2)]
    # Rough check the mass grid is wide enough to capture the posterior
    assert jnp.exp(lls_scaled.max()) / jnp.exp(lls_scaled.min()) > 100
    norm = jnp.trapz(jnp.exp(lls_scaled), M_chirp_MSUN_grid)

    print(norm)
    return jnp.sqrt(
        jnp.trapz(
            (M_chirp_MSUN_grid - M_chirp_MSUN) ** 2 * jnp.exp(lls_scaled),
            M_chirp_MSUN_grid,
        )
        / norm
    )


@click.command()
@click.option("--n_sig", default=25)
@click.option("--n_mach", default=25)
@click.option(
    "--sig_min", default=1e4, help="min value of sig0 in kg/m^2"
)
@click.option(
    "--sig_max", default=1e7, help="max value of sig0 in kg/m^2"
)
@click.option("--mach_min", default=10, help="min value of mach")
@click.option("--mach_max", default=30, help="max value of mach")
@click.option("--suffix", default="", help="suffix for output file")
def run(n_sig, n_mach, sig_min, sig_max, mach_min, mach_max, suffix):
    path = os.path.join("vacuum_fits_acc", f"vacuum_fits{suffix}.npz")
    sig_s = jnp.geomspace(
        sig_min, sig_max, n_sig
    )
    mach_s = jnp.linspace(mach_min, mach_max, n_mach)

    results = {
        k: np.full([n_sig, n_mach], np.nan)
        for k in [
            "snrs",
            "dN_naives",
            "matches",
            "dNs",
            "M_chirp_MSUN_bests",
            "M_chirp_MSUN_best_errs",
        ]
    }

    for i, sig in enumerate(tqdm(sig_s)):
        for j, mach in enumerate(mach_s):
            dd_a, f_l = setup_system_acc(sig, mach)
            fs = jnp.linspace(f_l, dd_a.f_c, 30)

            results["snrs"][i, j] = calculate_SNR(dd_a, fs)
    #        results["sigs"][i, j] = get_rho_s(rho_6, M_1_BM, gamma_s)

            # Dephasing relative to system with no DM
            dd_v = convert(dd_a, VacuumBinary)
            results["dN_naives"][i, j] = (Phi_to_c(f_l, dd_v) - jnp.real(Phi_to_c(f_l, dd_a)) )/ (2 * pi)

            # Don't waste effort on systems that are very hard to fit
#            if rho_6_to_rho6T(rho_6) < 5e-2:
            dd_v_best = fit_v(dd_a, f_l)
            results["M_chirp_MSUN_bests"][i, j] = dd_v_best.M_chirp / MSUN

            fs = jnp.linspace(f_l, dd_a.f_c, 100)
            pad_low, pad_high = get_match_pads(fs)
            results["matches"][i, j] = calculate_match_unnormd_fft(
                dd_v_best, dd_a, fs, pad_low, pad_high
            )

            results["dNs"][i, j] = (
                Phi_to_c(f_l, dd_v_best) - jnp.real(Phi_to_c(f_l, dd_a))
            ) / (2 * pi)

            results["M_chirp_MSUN_best_errs"][i, j] = get_M_chirp_err(
                dd_v_best, dd_a, f_l
            )

    results = {k: jnp.array(v) for k, v in results.items()}
    print(results)

    jnp.savez(
        path,
        m_1=M_1_BM,
        m_2=M_2_BM,
        M_chirp_MSUN=get_M_chirp(M_1_BM, M_2_BM) / MSUN,
        sig_s=sig_s,
        mach_s=mach_s,
        **results,
    )
    print(f"Results saved to {path}")


if __name__ == "__main__":
    run()
