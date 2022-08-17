wimprates
=========

Differential rates of WIMP-nucleus scattering in the standard halo model, for liquid xenon detectors.

`https://github.com/JelleAalbers/wimprates`

[![Test package](https://github.com/JelleAalbers/wimprates/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/JelleAalbers/wimprates/actions/workflows/pytest.yml)[![DOI](https://zenodo.org/badge/117823144.svg)](https://zenodo.org/badge/latestdoi/117823144)
[![Test package](https://github.com/JelleAalbers/wimprates/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/JelleAalbers/wimprates/actions/workflows/pytest.yml)
[![Coverage Status](https://coveralls.io/repos/github/JelleAalbers/wimprates/badge.svg?branch=master)](https://coveralls.io/github/JelleAalbers/wimprates?branch=master)

Installation and usage
----------------------
 - `pip install wimprates`
 - [See this basic example for usage.](https://github.com/JelleAalbers/wimprates/blob/master/notebooks/Example.ipynb)

The package uses numericalunits (https://pypi.python.org/pypi/numericalunits); all function inputs
are expected to have proper units.


Features
--------
- Spin-indendent and spin-dependent DM-nucleus scattering;
- Elastic NR, bremsstrahlung, and Migdal effect detection mechanisms;
- Time dependent observed dark matter speed distribution (annual modulation only, no daily modulation).
- Under development: DM-electron scattering


How to cite
------------
- J. Aalbers, B. Pelssers and K. Morå, wimprates: v0.3.0 (2019). http://doi.org/10.5281/zenodo.3345959.
- The original sources for models used in wimprates are:
  - **Spin-dependent scattering**: Klos, P. et al., [Phys.Rev. D88 (2013) no.8, 083516](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.88.083516), Erratum: [Phys.Rev. D89 (2014) no.2, 029901](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.89.029901). [[arXiv:1304.7684]](https://arxiv.org/abs/1304.7684)  
  - **Bremsstrahlung**: C. Kouvaris and J. Pradler, [Phys. Rev. Lett. 118, 031803 (2017)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.031803). [arXiv:1607.01789](https://arxiv.org/abs/1607.01789)
  - **Migdal effect**: M. Ibe et al., [JHEP 1803 (2018) 194](https://link.springer.com/article/10.1007/JHEP03(2018)194). [arXiv:1707.07258](https://arxiv.org/abs/1707.07258) 
  - **Dark matter electron scattering**: R. Essig, T. Volansky, T.-T. Yu: [Phys. Rev. D 96, 043017 (2017)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.043017). [arXiv:1703.00910](https://arxiv.org/abs/1703.00910).
    - Ionization form factors from: T.-T. Yu, http://ddldm.physics.sunysb.edu/ddlDM/, 2018-11-05.

Contributors
-------------
 * Jelle Aalbers
 * Bart Pelssers
 * Knut Dundas Mora
