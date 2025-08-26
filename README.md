# desisky

[![PyPI - Version](https://img.shields.io/pypi/v/desisky.svg)](https://pypi.org/project/desisky)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/desisky.svg)](https://pypi.org/project/desisky)

-----

## About

`desisky` provides two components for DESI sky modeling:

1. A **generative sky-spectrum model** that synthesizes realistic night-sky emission spectra conditioned on observational metadata, and
2. A **predictive broadband model** that returns surface brightness in the V, g, r, and z bands.

Built with **JAX/Equinox** and designed to integrate with SpecSim and survey forecasting workflows. This repository hosts the code and notebooks supporting the forthcoming paper by Dowicz et al. (20XX).

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install desisky
```

## License

`desisky` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
