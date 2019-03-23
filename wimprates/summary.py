"""
Summary functions
"""
import numericalunits as nu

import wimprates as wr
export, __all__ = wr.exporter()


@export
def rate_wimp(es, mw, sigma_nucleon, interaction='SI',
              detection_mechanism='elastic_nr', m_med=float('inf'), t=None,
              **kwargs):
    """Differential rate per unit time, unit detector mass
    and unit recoil energy of WIMP-nucleus scattering.
    Use numericalunits to get variables in/out in the right units.

    :param es: Energy of recoil (for elastic_nr)
    or emitted photon(for bremsstrahlung and Migdal)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction. Options:
        'SI' for spin-independent scattering;
            equal coupling to protons and neutrons.
        'SD_n_xxx' for spin-dependent scattering
            n can be 'n' or 'p' for neutron or proton coupling (at first order)
            x can be 'central', 'up' or 'down' for
                theoretical uncertainty on structure function.
    :param detection_mechanism: Detection mechanism, can be
         'elastic_nr' for regular elastic nuclear recoils
         'bremsstrahlung' for Bremsstrahlung photons
         'migdal' for the Migdal effect
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp.
    If not given, conservative velocity distribution is used.
    :param progress_bar: if True, show a progress bar during evaluation
    for multiple energies (if es is an array)

    :returns: numpy array of same length as es with
    differential WIMP-nucleus scattering rates.

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    dmechs = dict(elastic_nr=wr.rate_elastic,
                  bremsstrahlung=wr.rate_bremsstrahlung,
                  migdal=wr.rate_migdal)
    if detection_mechanism not in dmechs:
        raise NotImplementedError(
            "Unsupported detection mechanism '%s'" % detection_mechanism)
    return dmechs[detection_mechanism](
        es, mw=mw, sigma_nucleon=sigma_nucleon, interaction=interaction,
        m_med=m_med, t=t, **kwargs)


@export
def rate_wimp_std(es, mw, sigma_nucleon, m_med=float('inf'), t=None, **kwargs):
    """Differential rate per (ton year keV) of WIMP-nucleus scattering.
    :param es: Recoil energies in keV
    :param mw: WIMP mass in GeV/c^2
    :param sigma_nucleon: WIMP-nucleon cross-section in cm^2
    :param m_med: Medator mass in GeV/c^2. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp. If not given,
    conservative velocity distribution is used.
    :returns: numpy array of same length as es

    Further arguments are as for rate_wimp; see docstring of rate_wimp.
    """
    return (rate_wimp(es=es * nu.keV,
                      mw=mw * nu.GeV/nu.c0**2,
                      sigma_nucleon=sigma_nucleon * nu.cm**2,
                      m_med=m_med * nu.GeV/nu.c0**2, t=t, **kwargs)
            * (nu.keV * (1000 * nu.kg) * nu.year))
