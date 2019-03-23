"""Elastic nuclear recoil detected through Bremsstrahlung

Kouvaris/Pradler [arxiv:1607.01789v2]
"""
import numericalunits as nu
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

import wimprates as wr
export, __all__ = wr.exporter()


# Load the X-ray form factor
def to_itp(fn):
    x, y = np.loadtxt(wr.data_file('bs/' + fn), delimiter=',').T
    return interp1d(x, y, fill_value='extrapolate')


f1 = to_itp('atomic_form_1')
f2 = to_itp('atomic_form_2')


def vmin_w(w, mw):
    """Minimum wimp velocity to emit a Bremsstrahlung photon w

    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass

    From Kouvaris/Pradler [arxiv:1607.01789v2], equation in text below eq. 10
    """
    return (2 * w / wr.mu_nucleus(mw))**0.5


def erec_bound(sign, w, v, mw):
    """Bremsstrahlung scattering recoil energy kinematic limits
    From Kouvaris/Pradler [arxiv:1607.01789v2], eq. between 8 and 9,
    simplified by vmin (see above)

    :param sign: +1 to get upper limit, -1 to get lower limit
    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass
    :param v: WIMP speed (earth/detector frame)
    """
    return (wr.mu_nucleus(mw)**2 * v**2 / wr.mn()
            * (1
               - vmin_w(w, mw)**2 / (2 * v**2)
               + sign * (1 - vmin_w(w, mw)**2 / v**2)**0.5))


def sigma_w_erec(w, erec, v, mw, sigma_nucleon,
                 interaction='SI', m_med=float('inf')):
    """Differential WIMP-nucleus Bremsstrahlung cross section.
    From Kouvaris/Pradler [arxiv:1607.01789v2], eq. 8

    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass
    :param erec: recoil energy
    :param v: WIMP speed (earth/detector frame)
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    Default is 'SI' (spin-independent)
    :param m_med: Mediator mass. If not given, assumed very heavy.

    TODO: check for wmax!    # What is this? Still relevant?
    """
    # X-ray form factor
    form_atomic = np.abs(f1(w / nu.keV) + 1j * f2(w / nu.keV))

    # Note mn -> mn c^2, Kouvaris/Pradtler and McCabe use natural units
    return (4 * nu.alphaFS / (3 * np.pi * w) *
            erec / (wr.mn() * nu.c0**2) *
            form_atomic**2 *
            wr.sigma_erec(erec, v, mw, sigma_nucleon, interaction, m_med))


def sigma_w(w, v, mw, sigma_nucleon,
            interaction='SI', m_med=float('inf'), **kwargs):
    """Differential Bremsstrahlung WIMP-nucleus cross section

    :param w: Bremsstrahlung photon energy
    :param v: WIMP speed (earth/detector frame)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    Default is 'SI' (spin-independent)
    :param m_med: Mediator mass. If not given, assumed much heavier than mw.

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).

    """
    def integrand(erec):
        return sigma_w_erec(w, erec, v, mw, sigma_nucleon, interaction, m_med)

    return quad(integrand,
                erec_bound(-1, w, v, mw),
                erec_bound(+1, w, v, mw),
                **kwargs)[0]


@export
@wr.vectorize_first
def rate_bremsstrahlung(w, mw, sigma_nucleon, interaction='SI',
                        m_med=float('inf'), t=None, **kwargs):
    """Differential rate per unit detector mass and recoil energy of
    Bremsstrahlung elastic WIMP-nucleus scattering.

    :param w: Bremsstrahlung photon energy
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp. If not given,
    a conservative velocity distribution is used.
    :param interaction: string describing DM-nucleus interaction.
    See sigma_erec for options
    :param progress_bar: if True, show a progress bar during evaluation
    (if w is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    vmin = vmin_w(w, mw)

    if vmin >= wr.v_max(t):
        return 0

    def integrand(v):
        return (sigma_w(w, v, mw, sigma_nucleon, interaction, m_med) *
                v * wr.observed_speed_dist(v, t))

    return wr.rho_dm() / mw * (1 / wr.mn()) * quad(
        integrand,
        vmin,
        wr.v_max(t),
        **kwargs)[0]


# TODO: change to dblquad instead of 2x single quad!
