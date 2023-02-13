.. :changelog:

History
-------

0.5.0 (2023-02-13)
------------------
 * Change default v_0 and v_pec to match [current conventions](https://arxiv.org/abs/2105.00599) (#14)
 * If no time is provided, spectra are now calculated at a reference time (#14)
 * Fix bug where user-specified halo models would not override v_0 (#14)
 * Fix tests for numpy 1.24 (#15)

0.4.1 (2022-09-01)
------------------
 * Restore python 3.7 compatibility (#13)

0.4.0 (2022-08-14)
------------------
 * Fixes for alternate materials (#7)
 * Update notebooks, continuous integration tests (#9)

0.3.2 (2019-11-24)
------------------
* Fix technical release issue

0.3.1 (2019-11-24)
------------------
* Alternate materials for SI scattering (#4)
* Faster J200 timestamp conversion (#5)

0.3.0 (2019-07-22)
------------------
* Flexible halo model (#3)
* DM form factor choice for DM-electron scattering

0.2.2 (2019-03-27)
------------------
* DM-electron scattering

0.2.1 (2019-03-23)
------------------
* Fix package data specification

0.2.0 (2019-03-23)
------------------
* Annual modulation (#2)
* Migdal effect
* Configurable mediator mass
* Resistance to numericalunits.reset_units()
* Restructure as python package

0.1 (2018-01-17)
----------------
* Initial release
