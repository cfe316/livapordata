[![DOI](https://zenodo.org/badge/168408921.svg)](https://zenodo.org/badge/latestdoi/168408921)

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

Installation
============

This package can be installed from `PyPi`, using pip:

`pip install livapordata`

Citing
======

If you use this package in your research, please cite it (via Zenodo; link above).
