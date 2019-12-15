"""Migdal effect

"""
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import dblquad

import wimprates as wr
export, __all__ = wr.exporter()
# __all__ += ['BINDING_ENERGIES_MIGDAL']

def read_migdal_transitions(material='Xe'):
    # Differential transition probabilities for Xe vs energy (eV)

    df_migdal_material = pd.read_csv(wr.data_file('migdal/migdal_transition_%s.csv' %material))

    # Relevant (n, l) electronic states
    migdal_states_material = df_migdal_material.columns.values.tolist()
    migdal_states_material.remove('E')

    # Binding energies of the relevant Xenon electronic states
    # From table II of 1707.07258
    energy_nl = dict(
        Xe=np.array([3.5e4,
                     5.4e3, 4.9e3,
                     1.1e3, 9.3e2, 6.6e2,
                     2.0e2, 1.4e2, 6.1e1,
                     2.1e1, 9.8]),
        Ar=np.array([3.2e3,
                     3.0e2, 2.4e2,
                     2.7e1, 1.3e1]),
        Ge=np.array([1.1e4,
                     1.4e3, 1.2e3,
                     1.7e2, 1.2e2, 3.5e1,
                     1.5e1, 6.5e0])
    )

    binding_es_for_migdal_material = dict(zip(
        migdal_states_material, energy_nl[material]))

    consider_shells = dict(
        Xe=['3', '4'],
        # TODO, what are realistic values for Ge and Ar
        # TODO double check that this works for 2 as well and does not refer to 1_**2** or so
        # TODO, where is an argon migdal paper?
        Ar=['2'],
        Ge=['3'] #EDELWEIS
    )

    return df_migdal_material, binding_es_for_migdal_material, consider_shells[material]

# # Differential transition probabilities for Xe vs energy (eV)
# df_migdal = pd.read_csv(wr.data_file('migdal/migdal_transition_ps.csv'))
#
# # Relevant (n, l) electronic states
# migdal_states = df_migdal.columns.values.tolist()
# migdal_states.remove('E')
#
# # Binding energies of the relevant Xenon electronic states
# # From table II of 1707.07258

# binding_es_for_migdal = dict(zip(
#     migdal_states,
#     np.array([3.5e4,
#               5.4e3, 4.9e3,
#               1.1e3, 9.3e2, 6.6e2,
#               2e2, 1.4e2, 6.1e1,
#               2.1e1, 9.8])))

def vmin_migdal(w, erec, mw, material='Xe'):
    """Return minimum WIMP velocity to make a Migdal signal with energy w,
    given elastic recoil energy erec and WIMP mass mw.
    """
    y = (wr.mn(material) * erec / (2 * wr.mu_nucleus(mw) ** 2))**0.5
    y += w / (2 * wr.mn(material) * erec)**0.5
    return np.maximum(0, y)


@export
@wr.vectorize_first
def rate_migdal(w, mw, sigma_nucleon, interaction='SI', m_med=float('inf'),
                include_approx_nr=False, q_nr=0.15, material="Xe",
                t=None, halo_model=None, **kwargs):
    """Differential rate per unit detector mass and deposited ER energy of
    Migdal effect WIMP-nucleus scattering

    :param w: ER energy deposited in detector through Migdal effect
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    See sigma_erec for options
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param include_approx_nr: If True, instead return differential rate
        per *detected* energy, including the contribution of
        the simultaneous NR signal approximately, assuming q_{NR} = 0.15.
        This is how https://arxiv.org/abs/1707.07258
        presented the Migdal spectra.
    :param q_nr: conversion between Enr and Eee (see p. 27 of
        https://arxiv.org/pdf/1707.07258.pdf)
    :param material: name of the detection material (default is 'Xe')
    :param t: A J2000.0 timestamp.
    If not given, conservative velocity distribution is used.
    :param halo_model: class (default to standard halo model)
    containing velocity distribution
    :param progress_bar: if True, show a progress bar during evaluation
    (if w is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model
    include_approx_nr = 1 if include_approx_nr else 0

    result = 0
    df_migdal, binding_es_for_migdal, consider_shells = read_migdal_transitions(material=material)
    for state, binding_e in binding_es_for_migdal.items():
        binding_e *= nu.eV
        # Only consider n=3 and n=4
        # n=5 is the valence band so unreliable in in liquid
        # n=1,2 contribute very little
        if state[0] not in consider_shells:
            continue

        # Lookup for differential probability (units of ev^-1)
        p = interp1d(df_migdal['E'].values * nu.eV,
                     df_migdal[state].values / nu.eV,
                     bounds_error=False,
                     fill_value=0)

        def diff_rate(v, erec):
            # Observed energy = energy of emitted electron
            #                 + binding energy of state
            eelec = w - binding_e - include_approx_nr * erec * q_nr
            if eelec < 0:
                # print(w , binding_e , include_approx_nr * erec * 0.15)
                return 0

            return (
                # Usual elastic differential rate,
                # common constants follow at end
                wr.sigma_erec(erec, v, mw, sigma_nucleon, interaction,
                              m_med=m_med, material = material)
                * v * halo_model.velocity_dist(v, t)

                # Migdal effect |Z|^2
                # TODO: ?? what is explicit (eV/c)**2 doing here?
                * (nu.me * (2 * erec / wr.mn(material))**0.5 / (nu.eV / nu.c0))**2
                / (2 * np.pi)
                * p(eelec))

        # Note dblquad expects the function to be f(y, x), not f(x, y)...
        r = dblquad(
            diff_rate,
            0,
            wr.e_max(mw, wr.v_max(t, halo_model.v_esc)),
            lambda erec: vmin_migdal(w - include_approx_nr * erec * q_nr,
                                     erec, mw),
            lambda _: wr.v_max(t, halo_model.v_esc),
            **kwargs)[0]

        result += r

    return halo_model.rho_dm / mw * (1 / wr.mn(material)) * result
