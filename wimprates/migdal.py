"""
Migdal effect

Two implemented models:
 * Ibe et al: https://arxiv.org/abs/1707.07258
 * Cox et al: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.035032

 In the energy range of DM, the dipole approximation model implemented by Ibe et al
 is compatible with the one developped by Cox et al (check discussion in Cox et al)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Self

from fnmatch import fnmatch
from functools import lru_cache
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.integrate import dblquad
from scipy.interpolate import interp1d

import wimprates as wr


export, __all__ = wr.exporter()


@dataclass
class Shell:
    """
    Describes a specific atomic shell for the selected atom.

    Attributes:
        name (str): The name of the shell.
        element (str): The element class of the atom.
        binding_e (float): The binding energy for the shell.
        model (str): The model used for the single ionization probability computation.
        single_ionization_probability (Callable): A function to assign interpolators to.
            The interpolator will provide the single ionization probability for the shell
            according to the selected model.

    Methods:
        __call__(*args, **kwargs) -> np.ndarray:
            Calls the single_ionization_probability function with the given arguments and keyword arguments.

    Properties:
        n (int): Primary quantum number.
        l (str): Azimuthal quantum number for Ibe; Azimuthal + magnetic quantum number for Cox.
    """
    name: str
    element: str
    binding_e: float
    model: str
    single_ionization_probability: Callable  # to assign interpolators to

    def __call__(self: Self, *args, **kwargs) -> np.ndarray:
        return self.single_ionization_probability(*args, **kwargs)

    @property
    def n(self: Self) -> int:
        return int(self.name[0])

    @property
    def l(self: Self) -> str:
        return self.name[1:]


def _default_shells(material: str) -> list[str]:
    """
    Returns the default shells to consider for a given material.
    Args:
        material (str): The material for which to determine the default shells.
    Returns:
        list[str]: The default shells to consider for the given material.
    """

    consider_shells = dict(
        # For Xe, only consider n=3 and n=4
        # n=5 is the valence band so unreliable in liquid
        # n=1,2 contribute very little
        Xe=["3*", "4*"],
        # TODO, what are realistic values for Ar?
        Ar=["2*"],
        # EDELWEIS
        Ge=["3*"],
        Si=["2*"],
    )
    return consider_shells[material]


def create_cox_probability_function(
    element,
    state: str,
    material: str,
    dipole: bool = False,
) -> Callable[..., np.ndarray[Any, Any]]:
    
    fn_name = "dpI1dipole" if dipole else "dpI1"
    fn = getattr(element, fn_name)

    def get_probability(
        e: float | np.ndarray,  # energy of released electron
        erec: Optional[float | np.ndarray] = None,  # recoil energy
        v: Optional[float | np.ndarray] = None,  # recoil speed
    ) -> np.ndarray:
        if erec is None:
            if v is None:
                raise ValueError("Either v or erec have to be provided")
        elif v is None:
            v = (2 * erec / wr.mn(material)) ** 0.5 / nu.c0
        else:
            raise ValueError("Either v or erec have to be provided")
        
        e /= nu.keV

        input_points = wr.pairwise_log_transform(e, v)
        return fn(input_points, state) / nu.keV  # type: ignore

    return get_probability


@export
def get_migdal_transitions_probability_iterators(
    material: str = "Xe",
    model: str = "Ibe",
    considered_shells: Optional[list[str] | str] = None,
    dark_matter: bool = True,
    e_threshold: Optional[float] = None,
    dipole: bool = False,
    **kwargs,
) -> list[Shell]:
    # Differential transition probabilities for <material> vs energy (eV)

    # Check if considered_shells is an empty list
    if considered_shells is None:
        considered_shells = _default_shells(material)

    shells = []
    if model == "Ibe":
        df_migdal_material = pd.read_csv(
            wr.data_file("migdal/Ibe/migdal_transition_%s.csv" % material)
        )

        # Relevant (n, l) electronic states
        migdal_states_material = df_migdal_material.columns.values.tolist()
        migdal_states_material.remove("E")

        # Binding energies of the relevant electronic states
        # From table II of 1707.07258
        energy_nl = dict(
            Xe=np.array(
                [
                    3.5e4,
                    5.4e3,
                    4.9e3,
                    1.1e3,
                    9.3e2,
                    6.6e2,
                    2.0e2,
                    1.4e2,
                    6.1e1,
                    2.1e1,
                    9.8,
                ]
            ),
            Ar=np.array([3.2e3, 3.0e2, 2.4e2, 2.7e1, 1.3e1]),
            Ge=np.array([1.1e4, 1.4e3, 1.2e3, 1.7e2, 1.2e2, 3.5e1, 1.5e1, 6.5e0]),
            # http://www.chembio.uoguelph.ca/educmat/atomdata/bindener/grp14num.htm
            Si=np.array([1844.1, 154.04, 103.71, 13.46, 8.1517]),
        )

        for state, binding_e in zip(migdal_states_material, energy_nl[material]):
            if not any(fnmatch(state, take) for take in considered_shells):
                continue
            binding_e *= nu.eV

            # Lookup for differential probability (units of ev^-1)
            p = interp1d(
                np.array(df_migdal_material["E"].values) * nu.eV,
                df_migdal_material[state].values / nu.eV,
                bounds_error=False,
                fill_value=0,
            )

            shells.append(Shell(state, material, binding_e, model, p))

    elif model == "Cox":
        element = wr.cox_migdal_model(
            material,
            dipole=dipole,
            dark_matter=dark_matter,
            e_threshold=e_threshold,
        )

        for state, binding_e in element.orbitals:
            if not any(fnmatch(state, take) for take in considered_shells):
                continue

            shells.append(
                Shell(
                    state,
                    material,
                    binding_e * nu.keV,
                    model,
                    single_ionization_probability=create_cox_probability_function(
                        element,
                        state,
                        material,
                        dipole=dipole,
                    ),
                )
            )
    else:
        raise ValueError("Only 'Cox' and 'Ibe' models have been implemented")

    return shells


def vmin_migdal(
    w: np.ndarray, erec: np.ndarray, mw: float, material: str
) -> np.ndarray:
    """Return minimum WIMP velocity to make a Migdal signal with energy w,
    given elastic recoil energy erec and WIMP mass mw.
    """
    y = (wr.mn(material) * erec / (2 * wr.mu_nucleus(mw, material) ** 2)) ** 0.5
    y += w / (2 * wr.mn(material) * erec) ** 0.5
    return np.maximum(0, y)


@export
@wr.vectorize_first
def rate_migdal(
    w: np.ndarray,
    mw: float,
    sigma_nucleon: float,
    interaction: str = "SI",
    m_med: float = float("inf"),
    include_approx_nr: bool = False,
    q_nr: float = 0.15,
    material: str = "Xe",
    t: Optional[float] = None,
    halo_model: Optional[wr.StandardHaloModel] = None,
    consider_shells: Optional[list[str]] = None,
    migdal_model: str = "Ibe",
    dark_matter: bool = True,
    dipole: bool = False,
    e_threshold: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
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
    :param consider_shells: consider the following atomic shells, are
        fnmatched to the format from Ibe (i.e. 1_0, 1_1, etc).
    :param progress_bar: if True, show a progress bar during evaluation
    (if w is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model

    if not consider_shells:
        consider_shells = _default_shells(material)

    shells = get_migdal_transitions_probability_iterators(
        material=material,
        model=migdal_model,
        considered_shells=consider_shells,
        dipole=dipole,
        e_threshold=e_threshold,
        dark_matter=dark_matter,
    )

    result = 0
    for shell in shells:

        def diff_rate(v, erec):
            # Observed energy = energy of emitted electron
            #                 + binding energy of state
            eelec = w - shell.binding_e - include_approx_nr * erec * q_nr
            if eelec < 0:
                return 0

            if migdal_model == "Ibe":
                return (
                    # Usual elastic differential rate,
                    # common constants follow at end
                    wr.sigma_erec(
                        erec,
                        v,
                        mw,
                        sigma_nucleon,
                        interaction,
                        m_med=m_med,
                        material=material,
                    )
                    * v
                    * halo_model.velocity_dist(v, t)
                    # Migdal effect |Z|^2
                    # TODO: ?? what is explicit (eV/c)**2 doing here?
                    * (nu.me * (2 * erec / wr.mn(material)) ** 0.5 / (nu.eV / nu.c0))
                    ** 2
                    / (2 * np.pi)
                    * shell(eelec)
                )
            elif migdal_model == "Cox":
                return (
                    wr.sigma_erec(
                        erec,
                        v,
                        mw,
                        sigma_nucleon,
                        interaction,
                        m_med=m_med,
                        material=material,
                    )
                    * v
                    * halo_model.velocity_dist(v, t)
                    * shell(eelec, erec)
                )

        # Note dblquad expects the function to be f(y, x), not f(x, y)...
        r = dblquad(
            diff_rate,
            0,
            wr.e_max(mw, wr.v_max(t, halo_model.v_esc), wr.mn(material)),
            lambda erec: vmin_migdal(
                w=w - include_approx_nr * erec * q_nr,
                erec=erec,
                mw=mw,
                material=material,
            ),
            lambda _: wr.v_max(t, halo_model.v_esc),
            **kwargs,
        )[0]

        result += r

    return halo_model.rho_dm / mw * (1 / wr.mn(material)) * np.array(result)

@wr.deprecated("Use get_migdal_transitions_probability_iterators instead")
@lru_cache()
def read_migdal_transitions(material="Xe"):
    ### (DEPRECATED) Maintain this for backwards accessibility
    # Differential transition probabilities for <material> vs energy (eV)

    df_migdal_material = pd.read_csv(
        wr.data_file("migdal/Ibe/migdal_transition_%s.csv" % material)
    )

    # Relevant (n, l) electronic states
    migdal_states_material = df_migdal_material.columns.values.tolist()
    migdal_states_material.remove("E")

    # Binding energies of the relevant electronic states
    # From table II of 1707.07258
    energy_nl = dict(
        Xe=np.array(
            [3.5e4, 5.4e3, 4.9e3, 1.1e3, 9.3e2, 6.6e2, 2.0e2, 1.4e2, 6.1e1, 2.1e1, 9.8]
        ),
        Ar=np.array([3.2e3, 3.0e2, 2.4e2, 2.7e1, 1.3e1]),
        Ge=np.array([1.1e4, 1.4e3, 1.2e3, 1.7e2, 1.2e2, 3.5e1, 1.5e1, 6.5e0]),
        # http://www.chembio.uoguelph.ca/educmat/atomdata/bindener/grp14num.htm
        Si=np.array([1844.1, 154.04, 103.71, 13.46, 8.1517]),
    )

    binding_es_for_migdal_material = dict(
        zip(migdal_states_material, energy_nl[material])
    )

    return (
        df_migdal_material,
        binding_es_for_migdal_material,
    )
