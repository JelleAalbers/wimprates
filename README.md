wimprates
=========

`https://github.com/JelleAalbers/wimprates`

Differential rates of WIMP-nucleus scattering in the standard halo model.

Jelle Aalbers, 2018

Installation and usage
----------------------
 - Clone the repository and `cd` into its directory
 - `pip install -e .`
 - [See this basic example for usage.](https://github.com/JelleAalbers/wimprates/blob/master/notebooks/Example.ipynb)

Features
--------
  - Spin-indendent and spin-dependent DM-nucleus scattering
  - Bremsstrahlung and elastic NR detection modes

Limitations
-----------
 - Numeric integration is used to compute some differential rates, even in cases where exact expressions are known / could be derived.
 - The earth's motion w.r.t. to the sun is not taken into account: no annual modulation for you!
 - Not all functions are properly vectorized yet

The package uses numericalunits (https://pypi.python.org/pypi/numericalunits); all function inputs
are expected to have proper units. 

Do NOT call reset_units in your own code without reloading this module!
