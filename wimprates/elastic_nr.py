"""Elastic nuclear recoil rates, spin-independent and spin-dependent
"""

import numpy as np
import numericalunits as nu
from scipy.interpolate import interp1d
from scipy.integrate import quad

import wimprates as wr

export, __all__ = wr.exporter()
__all__ += ['ATOMIC_WEIGHT']

ATOMIC_WEIGHT = dict(
    Xe=131.293,
    Ar=39.948,
    Ge=72.64,
    Si=28.0855
)


@export
def mn(material='Xe'):
    """Mass of nucleus (not nucleon!)"""
    return ATOMIC_WEIGHT[material] * nu.amu


spin_isotopes = [
    # A, mass, J (nuclear spin), abundance
    # Data from Wikipedia (Jelle, 12 January 2018)
    (129, 128.9047794, 1/2, 26.401e-2),
    (131, 130.9050824, 3/2, 21.232e-2),
]

# Load spin-dependent structure functions
s_data = wr.load_pickle('sd/structure_f_erec_xe.pkl')
s_energies = s_data['_energies']
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
def mu_nucleus(mw, material):
    """DM-nucleus reduced mass"""
    return reduced_mass(mw, mn(material))


@export
def e_max(mw, v, m_nucleus):
    """Kinematic nuclear recoil energy maximum
    :param mw: Wimp mass
    :param m_nucleus: Nucleus mass. Defaults to standard atomic mass.
    :param v: Wimp speed
    """
    if m_nucleus is None:
        m_nucleus = mn()
    return 2 * reduced_mass(mw, m_nucleus)**2 * v**2 / m_nucleus


@export
def spherical_bessel_j1(x):
    """Spherical Bessel function j1 according to Wolfram Alpha"""
    return np.sin(x) / x**2 - np.cos(x) / x


@export
@wr.vectorize_first
def helm_form_factor_squared(erec, anucl):
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
    mnucl = nu.amu / (nu.GeV / nu.c0**2)  # Mass of a nucleon, in GeV/c^2
    pi = np.pi
    c = 1.23 * anucl**(1/3) - 0.60
    a = 0.52
    s = 0.9
    rn_sq = c**2 + (7.0/3.0) * pi**2 * a**2 - 5 * s**2
    rn = np.sqrt(rn_sq)  # units fm
    mass_kev = anucl * mnucl * 1e6
    hbarc_kevfm = 197327  # hbar * c in keV *fm (from Wolfram alpha)

    # E in units keV, rn in units fm, hbarc_kev units keV.fm
    # Formula is spherical bessel fn of Q=sqrt(E*2*Mn_keV)*rn
    q = np.sqrt(en * 2. * mass_kev)
    qrn_over_hbarc = q * rn / hbarc_kevfm
    sph_bess = spherical_bessel_j1(qrn_over_hbarc)
    retval = 9. * sph_bess * sph_bess / (qrn_over_hbarc * qrn_over_hbarc)
    qs_over_hbarc = q * s / hbarc_kevfm
    retval *= np.exp(-qs_over_hbarc * qs_over_hbarc)
    return retval


@export
def sigma_erec(erec, v, mw, sigma_nucleon,
               interaction='SI', m_med=float('inf'),
               material='Xe'):
    """Differential elastic WIMP-nucleus cross section
    (dependent on recoil energy and wimp-earth speed v)

    :param erec: recoil energy
    :param v: WIMP speed (earth/detector frame)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    See rate_wimps for options.
    :param m_med: Mediator mass. If not given, assumed much heavier than mw.
    :param material: name of the detection material (default is 'Xe')
    """
    if interaction == 'SI':
        sigma_nucleus = (sigma_nucleon
                         * (mu_nucleus(mw, material) / reduced_mass(
                            nu.amu, mw)) ** 2
                         * ATOMIC_WEIGHT[material]**2)
        result = (sigma_nucleus
                  / e_max(mw, v, mn(material))
                  * helm_form_factor_squared(erec,
                                             anucl=ATOMIC_WEIGHT[material]))

    elif interaction.startswith('SD'):
        if material != 'Xe':
            raise NotImplementedError("SD for %s-detectors not available" % (
                material))
        _, coupling, s_assumption = interaction.split('_')

        result = np.zeros_like(erec)
        for A, mn_isotope, J, abundance in spin_isotopes:
            mn_isotope *= nu.amu
            s = structure_functions[(A, coupling, s_assumption)]
            # x isn't quite sigma_nucleus:
            # you'd have to multilpy by structure(0),
            # then divide by it in the next line.
            # Obviously there's no point to this, so let's not.
            x = (sigma_nucleon * 4 * np.pi
                 * reduced_mass(mw, mn_isotope)**2
                 / (3 * reduced_mass(mw, nu.mp)**2 * (2 * J + 1)))
            result += (abundance
                       * x / e_max(mw, v, mn_isotope)
                       * s(erec / nu.keV))

    else:
        raise ValueError("Unsupported DM-nucleus interaction '%s'"
                         % interaction)

    return result * mediator_factor(erec, m_med, material)


@export
def mediator_factor(erec, m_med, material):
    if m_med == float('inf'):
        return 1
    q = (2 * mn(material) * erec)**0.5
    return m_med**4 / (m_med**2 + (q/nu.c0)**2) ** 2


@export
def vmin_elastic(erec, mw, material):
    """Minimum WIMP velocity that can produce a recoil of energy erec
    :param erec: recoil energy
    :param mw: Wimp mass
    :param material: name of the detection material (default is 'Xe')
    """
    return np.sqrt(mn(material) * erec / (2 * mu_nucleus(mw, material) ** 2))


@export
@wr.vectorize_first
def rate_elastic(erec, mw, sigma_nucleon, interaction='SI',
                 m_med=float('inf'), t=None, material='Xe',
                 halo_model=None, **kwargs):
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
    :param halo_model: class (default to standard halo model)
    containing velocity distribution
    :param progress_bar: if True, show a progress bar during evaluation
    (if erec is an array)
    :param material: name of the detection material (default is 'Xe')

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).

    Analytic expressions are known for this rate, but they are not used here.
    """

    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model
    v_min = vmin_elastic(erec, mw, material)

    if v_min >= wr.v_max(t, halo_model.v_esc):
        return 0

    def integrand(v):
        return (sigma_erec(erec, v, mw, sigma_nucleon,
                           interaction, m_med, material=material) * v
                * halo_model.velocity_dist(v, t))

    return halo_model.rho_dm / mw * (1 / mn(material)) * quad(
        integrand,
        v_min, wr.v_max(t, halo_model.v_esc),
        **kwargs)[0]
