import warnings

from datetime import datetime
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy import interpolate, integrate

import wimprates as wr
export, __all__ = wr.exporter()


@export
class HaloModel:
    """Base class for dark matter halos.

    Subclasses should override velocity_dist and v_max.

    This will then precompute the inverse mean speed for a range of minimum
    velocities.
    """

    def velocity_dist(self, v, t):
        """Return normalized speed distribution of dark matter f_v(v)
        in the Earth's rest frame. Units are those of inverse speed.

        :param v: dark matter speed
        :param t: J2000.0 timestamp or None
        """
        raise NotImplementedError

    def v_max(self, t):
        """Return maximum dark matter velocity

        :param t: J2000.0 timestamp or None
        """
        raise NotImplementedError

    def inverse_mean_speed(self, v_min, t=None):
        """Inverse mean dark matter speed above a cutoff v_min
        (i.e. the integral of 1/v * f(v) from v_min to v_max)

        :param v_min: Lower bound of the speed integral
        :param t: J2000.0 timestamp
        """
        if t is None:
            if not hasattr(self, '_inverse_mean_speed_kms'):
                self._build_ims_interpolator()
            # Use precomputed value
            return self._inverse_mean_speed_kms(v_min / (nu.km/nu.s)) / (nu.km/nu.s)
        else:
            # Compute on the fly
            return self._ims(v_min, t)

    def _ims(self, v_min, t):
        return integrate.quad(lambda v: 1 / v * self.velocity_dist(v,t),
            v_min, self.v_max(t))[0]

    def _build_ims_interpolator(self):
        """Build interpolator for inverse mean speed at the standard time,
        and store it in self._inverse_mean_speed_kms.
        """
        # Precompute inverse mean speed for a range of likely v_mins,
        # for t = None.
        # TODO: number of points is hardcoded, should be made configurable.
        _v_mins = np.linspace(0, 1, 2500) * self.v_max(t=None)
        _ims = np.array([
            self._ims(_v_min, None)
            for _v_min in _v_mins])

        # Store interpolator in km/s rather than unit-dependent numbers
        # so we don't have to recalculate them when nu.reset_units() is called
        self._inverse_mean_speed_kms = interpolate.interp1d(
            _v_mins / (nu.km/nu.s),
            _ims * (nu.km/nu.s),
            # If we don't have 0 < v_min < v_max, we want to return 0
            # so the integrand vanishes
            fill_value=0, bounds_error=False)


##
# Standard halo model
##

# See https://arxiv.org/abs/2105.00599 and references therein.
# These are declared without units, so users can safely do reset_units.
# Functions should apply the appropriate numericalunits prefactors
# when using these
_SHM_DEFAULTS = dict(
    rho_dm = 0.3,               # GeV / c^2 / cm^3
    v_esc = 544,                # km/s
    v_orbit = 29.8,             # km/s
    v_pec = (11.1, 12.2, 7.3),  # km/s
    v_0 = 238,                  # km/s
)

@export
class StandardHaloModel(HaloModel):
    """Standard halo model

    :param v_0: Local standard of rest velocity
    :param v_esc: Escape velocity
    :param rho_dm: Density in mass/volume dark matter at the Earth
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        # Store parameters in known units, so users can safely reset_units
        # after initializing halo models.
        self._v_0_kms = _SHM_DEFAULTS['v_0'] if v_0 is None else v_0 / (nu.km/nu.s)
        self._v_esc_kms = _SHM_DEFAULTS['v_esc'] if v_esc is None else v_esc / (nu.km/nu.s)
        self._rho_dm_gevc2cm3 = _SHM_DEFAULTS['rho_dm'] if rho_dm is None else rho_dm / (nu.GeV/nu.c0**2 / nu.cm**3)
        super().__init__()

    @property
    def v_0(self):
        return self._v_0_kms * nu.km/nu.s

    @property
    def v_esc(self):
        return self._v_esc_kms * nu.km/nu.s

    @property
    def rho_dm(self):
        return self._rho_dm_gevc2cm3 * nu.GeV/nu.c0**2 / nu.cm**3

    def velocity_dist(self, v, t=None):
        """Observed distribution of dark matter particle speeds on earth
        under the standard halo model.

        See my thesis for derivation ;-)
        If you find a paper where this formula is written out explicitly, please
        let me know. I spent a lot of time looking for this in vain.

        Optionally supply J2000.0 time t to take into account Earth's orbital
        velocity.

        Further inputs: scale velocity v_0 and escape velocity v_esc_value
        """
        v_0 = self.v_0
        v_esc = self.v_esc
        v_earth_t = v_earth(t, v_0=v_0)

        # Normalization constant, see Lewin&Smith appendix 1a
        _w = v_esc/v_0
        k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)  # unitless

        # Maximum cos(angle) for this velocity, otherwise v0
        xmax = np.minimum(1,
                        (v_esc**2 - v_earth_t**2 - v**2)
                        / (2 * v_earth_t * v))
        # unitless

        y = (k * v / (np.pi**0.5 * v_0 * v_earth_t)
            * (np.exp(-((v-v_earth_t)/v_0)**2)
            - np.exp(-1/v_0**2 * (v**2 + v_earth_t**2
                    + 2 * v * v_earth_t * xmax))))
        # units / (velocity)

        # Zero if v > v_max
        try:
            len(v)
        except TypeError:
            # Scalar argument
            if v > v_max_shm(t, v_esc, v_0=v_0):
                return 0
            else:
                return y
        else:
            # Array argument
            y[v > v_max_shm(t, v_esc, v_0=v_0)] = 0
            return y

    def v_max(self, t):
        """Maximum dark matter velocity under the standard halo model

        :param t: J2000.0 timestamp or None
        """
        if t is None:
            return self.v_esc + v_earth(t, v_0=self.v_0)
        else:
            return self.v_esc + np.sum(earth_velocity(t, v_0=self.v_0) ** 2) ** 0.5


# J2000.0 epoch conversion (converts datetime to days since epoch)
# Zero of this convention is defined as 12h Terrestrial time on 1 January 2000
# This is similar to UTC or GMT with negligible error (~1 minute).
# See http://arxiv.org/abs/1312.1355 Appendix A for more details
# Test case for 6pm GMT 31st January 2009
#  j2000(2009, 1, 31.75) = 3318.25
#  j2000(date=pd.to_datetime('2009-1-31 18:00:00') = 3318.25
@export
def j2000(date):
    """Returns the fractional number of days since J2000.0 epoch.
    Either pass:
      * An integer or array of integers (date argument), ns since unix epoch
      * datetime.datetime object
      * pd.Timestamp object
    Day of month starts at 1.
    """
    zero = pd.to_datetime('2000-01-01T12:00')
    nanoseconds_per_day = 1e9 * 3600 * 24
    if isinstance(date, datetime):
        # pd.datetime refers to datetime.datetime
        # make it into a pd.Timestamp
        # Timestamp.value gives timestamp in ns
        date = pd.to_datetime(date).value
    elif isinstance(date, pd.Timestamp):
        date = date.value
    return (date - zero.value) / nanoseconds_per_day


@export
def j2000_from_ymd(year, month, day_of_month):
    """"Returns the fractional number of days since J2000.0 epoch.
    :param year: Year
    :param month: Month (January = 1)
    :param day: Day of month (starting from 1), fractional days are
    relative to midnight UT.
    """
    assert month > 0
    assert month < 13

    y = year if month > 2 else year - 1
    m = month if month > 2 else month + 12

    return (np.floor(365.25 * y)
            + np.floor(30.61 * (m + 1))
            + day_of_month - 730563.5)


@export
def earth_velocity(t, v_0 = None):
    """Returns 3d velocity of earth, in the galactic rest frame,
    in galactic coordinates.
    :param t: J2000.0 timestamp
    :param v_0: Local standard of rest velocity

    Values and formula from https://arxiv.org/abs/1209.3339
    Assumes earth circular orbit.
    """
    if v_0 is None :
        v_0 = _SHM_DEFAULTS['v_0'] * nu.km/nu.s

    # e_1 and e_2 are the directions of earth's velocity at t1
    # and t1 + 0.25 year.
    e_1 = np.array([0.9931, 0.1170, -0.01032])
    e_2 = np.array([-0.0670, 0.4927, -0.8676])
    # t1 is the time of the vernal equinox, March 21. Does it matter what
    # year? Precession of equinox takes 25800 years so small effect.
    t1 = j2000_from_ymd(2000, 3, 21)
    # Angular frequency
    omega = 2 * np.pi / 365.25
    phi = omega * (t - t1)

    # Mean orbital velocity of the Earth (Lewin & Smith appendix B)
    v_orbit = _SHM_DEFAULTS['v_orbit'] * nu.km / nu.s

    v_earth_sun = v_orbit * (e_1 * np.cos(phi) + e_2 * np.sin(phi))

    # Velocity of Local Standard of Rest
    v_lsr = np.array([0, v_0, 0])
    # Solar peculiar velocity
    v_pec = np.array(_SHM_DEFAULTS['v_pec']) * nu.km/nu.s

    return v_lsr + v_pec + v_earth_sun


@export
def v_earth(t=None, v_0=None):
    """Return speed of earth relative to galactic rest frame
    Velocity of earth/sun relative to gal. center (eccentric orbit, so not
    equal to v_0).

    :param t: J2000 timestamp or None
    :param v_0: Local standard of rest velocity
    """
    if t is None:
        # This day (Feb 29 2000) gives ~ the annual average speed
        t = 59.37
    return np.sum(earth_velocity(t, v_0=v_0) ** 2) ** 0.5


@export
def v_max_shm(t=None, v_esc=None, v_0=None):
    """Return maximum observable dark matter velocity on Earth
    according to the standard halo model."""
    # defaults
    v_esc = _SHM_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
    v_0 = _SHM_DEFAULTS['v_0'] * nu.km / nu.s if v_0 is None else v_0
    # args do not change value when you do a
    # reset_unit so this is necessary to avoid errors
    if t is None:
        return v_esc + v_earth(t, v_0=v_0)
    else:
        return v_esc + np.sum(earth_velocity(t, v_0=v_0) ** 2) ** 0.5


@export
def v_max(t=None, v_esc=None, v_0=None):
    warnings.warn(
        "v_max is deprecated. Use wr.StandardHaloModel(v_0=.., v_esc=...).v_max(t)",
        DeprecationWarning)
    return wr.v_max_shm(t=t, v_esc=v_esc, v_0=v_0)


@export
def observed_speed_dist(v, t=None, v_0=None, v_esc=None):
    warnings.warn(
        "observed_speed_dist is deprecated. Use wr.StandardHaloModel(v_0=.., v_esc=...).velocity_dist(v, t)",
        DeprecationWarning)
    return wr.StandardHaloModel(v_0=v_0, v_esc=v_esc).velocity_dist(v, t)


# Preconstructed SHM instances so the inverse speed calculation does not trigger
# repeatedly.
STANDARD_HALO_MODEL = StandardHaloModel()
__all__ += ['STANDARD_HALO_MODEL']


##
# Models that provide a differential flux instead of a speed distribution
##

@export
class DifferentialFluxHaloModelWrapper(HaloModel):
    """Provides a 'halo model' for models with a known differential flux,
        number_density * v * velocity_dist(v)


    Subclasses should implement differential_flux and v_max.

    This model's velocity_dist gives the correct quantity to insert in a
    differential rate calculation that assumes halo dark matter, i.e.
        number_density = rho_dm / mw

    Thus, 'velocity_dist' returns
        differential_flux(v, t) * mw / (rho_dm * v)

    :param mw: Dark matter mass
    :param rho_dm: Dark matter mass density. If not provided,
        use the default value from the standard halo model.
    """

    def __init__(self, mw, rho_dm=None):
        # As in StandardHaloModel, store attributes in known units
        self._mw = mw / (nu.GeV/nu.c0**2)
        self._rho_dm = _SHM_DEFAULTS['rho_dm'] if rho_dm is None else rho_dm / (nu.GeV/nu.c0**2 / nu.cm**3)

    @property
    def mw(self):
        return self._mw * nu.GeV/nu.c0**2

    @property
    def rho_dm(self):
        return self._rho_dm * nu.GeV/nu.c0**2 / nu.cm**3

    def velocity_dist(self, v, t):
        return (
            self.differential_flux(v, t)
            * self.mw / (self.rho_dm * v))

    def differential_flux(self, v, t):
        raise NotImplementedError

    def vmax(self, t):
        raise NotImplementedError


srdm_fluxes = wr.load_pickle('srdm/consolidated_fluxes.pickle')['flux_bag']
# Sort the keys so users can more easiliy find available fluxes
srdm_fluxes = {k: srdm_fluxes[k] for k in sorted(srdm_fluxes)}


@export
class SolarReflectedDMEModel(DifferentialFluxHaloModelWrapper):
    """
    Model for solar reflected dark matter that experiences dark matter
    electron scattering.

    Fluxes are extracted from DAMASCUS-Sun, see 2102.12483v2.
    Only time-averaged fluxes are available; t arguments are ignored.

    :param mw: Dark matter mass
    :param sigma_dme: Dark matter-electron scattering cross-section as defined
        in wimprates.electron.rate_dme.
    :param rho_dm: Dark matter mass density. If not provided,
        use the default value from the standard halo model.
    """

    def __init__(self, mw, sigma_dme, rho_dm=None):
        super().__init__(mw, rho_dm=rho_dm)
        # We don't have to store sigma_dme, just need it once
        # to look up the right flux here.

        # Look up fluxes for this mass and cross-section
        xsec_cm2 = sigma_dme / nu.cm**2
        mass_gevc2 = self.mw / (nu.GeV/nu.c0**2)
        key = f'mass{mass_gevc2:.3e}_xsec{xsec_cm2:.3e}'
        try:
            df = srdm_fluxes[key]
        except KeyError:
            raise ValueError(
                "No available flux for this mass and cross-section. "
                "See list(wimprates.halo.srdm_fluxes) for available values; "
                f"we did not find {key}.")

        # Construct interpolator for the flux.
        # 'Speed' is in units of km/s, 'Differential Flux' is in units of
        # events / (km/s) / cm2 / s.
        # As usual, however, we must store attributes in fixed units.

        self._differential_flux = interpolate.interp1d(
            df['Speed'].values,
            df['Differential Flux'].values,
            kind='linear',
            fill_value=0, bounds_error=False)

        self._v_max_kms = df['Speed'].values.max()

    def v_max(self, t=None):
        return self._v_max_kms * nu.km/nu.s

    def differential_flux(self, v, t=None):
        v = v / (nu.km/nu.s)
        diff_flux = self._differential_flux(v)
        return diff_flux * (1 / (nu.km/nu.s) / nu.cm**2 / nu.s)
