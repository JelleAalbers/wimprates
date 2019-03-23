"""Elastic nuclear recoil rates, spin-independent and spin-dependent
"""

import numpy as np
import numericalunits as nu
from scipy.interpolate import interp1d
from scipy.integrate import quad

import wimprates as wr
export, __all__ = wr.exporter()
__all__ += 'An mn'.split()

# Standard atomic weight of target (averaged across all isotopes)
An = 131.293
mn = An * nu.amu    # Mass of nucleus (not nucleon!)

spin_isotopes = [
    # A, mass, J (nuclear spin), abundance
    # Data from Wikipedia (Jelle, 12 January 2018)
    (129, 128.9047794 * nu.amu, 1/2, 26.401e-2),
    (131, 130.9050824 * nu.amu, 3/2, 21.232e-2),
]

# Load spin-dependent structure functions
s_data = wr.load_pickle('sd/structure_f_erec_xe.pkl')
s_energies = s_data['_energies'] * nu.keV
structure_functions = {}
# Don't use k, v; dangerous globals...
for _k, _v in s_data.items():
    if _k == '_energies':
        continue
    structure_functions[_k] = interp1d(s_energies, _v,
                                       bounds_error=False, fill_value=0)


@export
def reduced_mass(m1, m2):
    return m1 * m2 / (m1 + m2)


@export
def e_max(mw, v, mn=mn):
    """Kinematic nuclear recoil energy maximum
    :param mw: Wimp mass
    :param mn: Nucleus mass. Defaults to standard atomic mass.
    :param v: Wimp speed
    """
    return 2 * reduced_mass(mw, mn)**2 * v**2 / mn


@export
def spherical_bessel_j1(x):
    """Spherical Bessel function j1 according to Wolfram Alpha"""
    return np.sin(x)/x**2 + - np.cos(x)/x


@export
@wr.vectorize_first
def helm_form_factor_squared(erec, anucl=An):
    """Return Helm form factor squared from Lewin & Smith

    Lifted from Andrew Brown's code with minor edits

    :param erec: nuclear recoil energy
    :param anucl: Nuclear mass number
    """
    en = erec / nu.keV
    if anucl <= 0:
        raise ValueError("Invalid value of A!")

    # TODO: Rewrite this so it doesn't use its internal unit system
    #  and hardcoded constants...

    # First we get rn squared, in fm
    mnucl = nu.amu/(nu.GeV/nu.c0**2)    # Mass of a nucleon, in GeV/c^2
    pi = np.pi
    c = 1.23*anucl**(1/3)-0.60
    a = 0.52
    s = 0.9
    rn_sq = c**2 + (7.0/3.0) * pi**2 * a**2 - 5 * s**2
    rn = np.sqrt(rn_sq)  # units fm
    mass_kev = anucl * mnucl * 1e6
    hbarc_kevfm = 197327  # hbar * c in keV *fm (from Wolfram alpha)

    # E in units keV, rn in units fm, hbarc_kev units keV.fm
    # Formula is spherical bessel fn of Q=sqrt(E*2*Mn_keV)*rn
    q = np.sqrt(en*2.*mass_kev)
    qrn_over_hbarc = q*rn/hbarc_kevfm
    sph_bess = spherical_bessel_j1(qrn_over_hbarc)
    retval = 9. * sph_bess * sph_bess / (qrn_over_hbarc*qrn_over_hbarc)
    qs_over_hbarc = q*s/hbarc_kevfm
    retval *= np.exp(-qs_over_hbarc*qs_over_hbarc)
    return retval


@export
def sigma_erec(erec, v, mw, sigma_nucleon,
               interaction='SI', m_med=float('inf')):
    """Differential elastic WIMP-nucleus cross section
    (dependent on recoil energy and wimp-earth speed v)

    :param erec: recoil energy
    :param v: WIMP speed (earth/detector frame)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    See rate_wimps for options.
    :param m_med: Mediator mass. If not given, assumed much heavier than mw.
    """
    if interaction == 'SI':
        sigma_nucleus = (sigma_nucleon
                         * (reduced_mass(mn, mw) / reduced_mass(nu.amu, mw))**2
                         * An**2)
        result = (sigma_nucleus
                  / e_max(mw, v)
                  * helm_form_factor_squared(erec, anucl=An))

    elif interaction.startswith('SD'):
        _, coupling, s_assumption = interaction.split('_')

        result = np.zeros_like(erec)
        for A, mn_isotope, J, abundance in spin_isotopes:
            s = structure_functions[(A, coupling, s_assumption)]
            # x isn't quite sigma_nucleus:
            # you'd have to multilpy by structure(0),
            # then divide by it in the next line.
            # Obviously there's no point to this, so let's not.
            x = (sigma_nucleon * 4 * np.pi
                 * reduced_mass(mw, mn_isotope)**2
                 / (3 * reduced_mass(mw, nu.mp)**2 * (2 * J + 1)))
            result += abundance * x / e_max(mw, v, mn=mn_isotope) * s(erec)

    else:
        raise ValueError("Unsupported DM-nucleus interaction '%s'"
                         % interaction)

    return result * mediator_factor(erec, m_med)


@export
def mediator_factor(erec, m_med):
    if m_med == float('inf'):
        return 1
    q = (2 * mn * erec)**0.5
    return m_med**4 / (m_med**2 + (q/nu.c0)**2)**2


@export
def vmin_elastic(erec, mw):
    """Minimum WIMP velocity that can produce a recoil of energy erec
    :param erec: recoil energy
    :param mw: Wimp mass
    """
    return np.sqrt(mn * erec / (2 * reduced_mass(mw, mn)**2))


@export
@wr.vectorize_first
def rate_elastic(erec, mw, sigma_nucleon, interaction='SI', m_med=float('inf'),
                 t=None, **kwargs):
    """Differential rate per unit detector mass and recoil energy of
    elastic WIMP scattering

    :param erec: recoil energy
    :param mw: WIMP mass
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction,
    see sigma_erec for options
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp.
    If not given, conservative velocity distribution is used.
    :param progress_bar: if True, show a progress bar during evaluation
    (if erec is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).

    Analytic expressions are known for this rate, but they are not used here.
    """
    v_min = vmin_elastic(erec, mw)

    if v_min >= wr.v_max(t):
        return 0

    def integrand(v):
        return (sigma_erec(erec, v, mw, sigma_nucleon, interaction, m_med)
                * v * wr.observed_speed_dist(v, t))

    return wr.rho_dm / mw * (1 / mn) * quad(
        integrand,
        v_min, wr.v_max(t),
        **kwargs)[0]
