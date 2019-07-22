"""Dark matter - electron scattering
"""
import numericalunits as nu
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import quad, dblquad

import wimprates as wr
export, __all__ = wr.exporter()
__all__ += ['dme_shells', 'l_to_letter', 'l_to_number']

# Load form factor and construct interpolators
shell_data = wr.load_pickle('dme/dme_ionization_ff.pkl')
for _shell, _sd in shell_data.items():
    _sd['log10ffsquared_itp'] = RegularGridInterpolator(
        (_sd['lnks'], _sd['lnqs']),
        np.log10(_sd['ffsquared']),
        bounds_error=False, fill_value=-float('inf'),)


dme_shells = [(5, 1), (5, 0), (4, 2), (4, 1), (4, 0)]
l_to_number = dict(s=0, p=1, d=2, f=3)
l_to_letter = {v: k for k, v in l_to_number.items()}


@export
def shell_str(n, l):
    if isinstance(l, str):
        return str(n) + l
    return str(n) + l_to_letter[l]



@export
def dme_ionization_ff(shell, e_er, q):
    """Return dark matter electron scattering ionization form factor

    Outside the parametrized range, the form factor is assumed 0
    to give conservative results.

    :param shell: Name of atomic shell, e.g. '4p'
        Note not all shells are included in the data.
    :param e_er: Electronic recoil energy
    :param q: Momentun transfer
    """
    if isinstance(shell, tuple):
        shell = shell_str(*shell)

    lnq = np.log(q / (nu.me * nu.c0 * nu.alphaFS))
    # From Mathematica: (*ER*) (2 lnkvalues[[j]])/Log[10]
    # log10 (E/Ry) = 2 lnk / ln10
    # lnk = log10(E/Ry) * ln10 / 2
    #     = lng(E/Ry) / 2
    # Ry = rydberg = 13.6 eV
    ry = nu.me * nu.e ** 4 / (8 * nu.eps0 ** 2 * nu.hPlanck ** 2)
    lnk = np.log(e_er / ry) / 2
    return 10**(shell_data[shell]['log10ffsquared_itp'](
        np.vstack([lnk, lnq]).T))


@export
def binding_es_for_dme(n, l):
    """Return binding energy of Xenon's (n, l) orbital
    according to Essig et al. 2017 Table II

    Note these are different from e.g. Ibe et al. 2017!
    """

    return {'4s': 213.8,
            '4p': 163.5,
            '4d': 75.6,
            '5s': 25.7,
            '5p': 12.4}[shell_str(n, l)] * nu.eV


@export
def v_min_dme(eb, erec, q, mw):
    """Minimal DM velocity for DM-electron scattering
    :param eb: binding energy of shell
    :param erec: electronic recoil energy energy
    :param q: momentum transfer
    :param mw: DM mass
    """
    return (erec + eb) / q + q / (2 * mw)


# Precompute velocity integrals for t=None
@export 
def velocity_integral_without_time(halo_model=None):
    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model
    _v_mins = np.linspace(0, 1, 1000) * wr.v_max(None, halo_model.v_esc)
    _ims = np.array([
        quad(lambda v: 1 / v * halo_model.velocity_dist(v,None),
            _v_min,
             wr.v_max(None, halo_model.v_esc ))[0]
        for _v_min in _v_mins])
    
    # Store interpolator in km/s rather than unit-dependent numbers
    # so we don't have to recalculate them when nu.reset_units() is called
    inverse_mean_speed_kms = interp1d(
        _v_mins / (nu.km/nu.s),
        _ims * (nu.km/nu.s),
        # If we don't have 0 < v_min < v_max, we want to return 0
        # so the integrand vanishes
        fill_value=0, bounds_error=False)
    return inverse_mean_speed_kms

inverse_mean_speed_kms = velocity_integral_without_time()


@export
@wr.vectorize_first
def rate_dme(erec, n, l, mw, sigma_dme,
             f_dm='1',
             t=None, halo_model = None, **kwargs):
    """Return differential rate of dark matter electron scattering vs energy
    (i.e. dr/dE, not dr/dlogE)
    :param erec: Electronic recoil energy
    :param n: Principal quantum numbers of the shell that is hit
    :param l: Angular momentum quantum number of the shell that is hit
    :param mw: DM mass
    :param sigma_dme: DM-free electron scattering cross-section at fixed
    momentum transfer q=0
    :param f_dm: One of the following:
     '1':     |F_DM|^2 = 1, contact interaction / heavy mediator (default)
     '1_q':   |F_DM|^2 = (\alpha m_e c / q), dipole moment
     '1_q2': |F_DM|^2 = (\alpha m_e c / q)^2, ultralight mediator
    :param t: A J2000.0 timestamp.
    If not given, a conservative velocity distribution is used.
    :param halo_model: class (default to standard halo model) containing velocity distribution
    """
    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model
    shell = shell_str(n, l)
    eb = binding_es_for_dme(n, l)

    f_dm = {
        '1': lambda q: 1,
        '1_q': lambda q: nu.alphaFS * nu.me * nu.c0 / q,
        '1_q2': lambda q: (nu.alphaFS * nu.me * nu.c0 / q)**2
    }[f_dm]

    # No bounds are given for the q integral
    # but the form factors are only specified in a limited range of q
    qmax = (np.exp(shell_data[shell]['lnqs'].max())
            * (nu.me * nu.c0 * nu.alphaFS))

    if t is None:
        # Use precomputed inverse mean speed,
        # so we only have to do a single integral
        def diff_xsec(q):
            vmin = v_min_dme(eb, erec, q, mw)
            result = q * dme_ionization_ff(shell, erec, q) * f_dm(q)**2
            # Note the interpolator is in kms, not unit-carrying numbers
            # see above
            result *= inverse_mean_speed_kms(vmin / (nu.km/nu.s))
            result /= (nu.km/nu.s)
            return result

        r = quad(diff_xsec, 0, qmax)[0]

    else:
        # Have to do double integral
        # Note dblquad expects the function to be f(y, x), not f(x, y)...
        def diff_xsec(v, q):
            result = q * dme_ionization_ff(shell, erec, q) * f_dm(q)**2
            result *= 1 / v * halo_model.velocity_dist(v, t)
            return result

        r = dblquad(
            diff_xsec,
            0,
            qmax,
            lambda q: v_min_dme(eb, erec, q, mw),
            lambda _: wr.v_max(t, halo_model.v_esc),
            **kwargs)[0]

    mu_e = mw * nu.me / (mw + nu.me)

    return (
        # Convert cross-section to rate, as usual
        halo_model.rho_dm / mw * (1 / wr.mn())
        # d/lnE -> d/E
        * 1 / erec
        # Prefactors in cross-section
        * sigma_dme / (8 * mu_e ** 2)
        * r)
