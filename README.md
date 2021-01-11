[![DOI](https://zenodo.org/badge/168408921.svg)](https://zenodo.org/badge/latestdoi/168408921)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![PyPI](https://img.shields.io/pypi/v/livapordata)


Lithium vapor data
==================

This python module contains various data useful for physical modeling of lithium, especially neutral lithium vapor.

It has data on

- Vapor pressure
- Vapor viscosity
- Vapor thermal conductivity
- Vapor self-diffusivity

as functions of temperature, each from a variety of sources. If you find sources in the literature not listed here, I'd be interested to incorporate them!

It also contains functions to calculate

- Equilibrium vapor flux (Langmuir flux)
- Surface tension

And for convenience, provides basic data on

- (averaged) Li atomic mass
- Heat of vaporization

This module is most concerned with temperatures from 300 K to 1500 K.

Multiple sources from the literature are included wherever possible. (Nearly) All data sources are cited.

Usage
=====

```
>>> from livapordata import lithium
>>> li = lithium.LithiumProperties()
>>> t_k = 900 # Kelvin
>>> li.vapor_pressure(t_k) # in Pa
12.77074
>>> li.langmuir_flux(t_k) # in m^{-2} s^{-1}
4.257543e+23
```
Also see the example scripts provided, which plot the various lithium vapor pressure and viscosity functions.

Installation
============

This package can be installed from `PyPi`, using pip:

`pip install livapordata`

Citing
======

If you use this package in your research, please cite it (via Zenodo; link above).
