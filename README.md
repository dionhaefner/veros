<p align="center">
<img src="doc/_images/veros-logo-400px.png?raw=true">
</p>

<p align="center">
  <a href="http://veros.readthedocs.io/?badge=latest">
    <img src="https://readthedocs.org/projects/veros/badge/?version=latest" alt="Documentation status">
  </a>
  <a href="https://travis-ci.org/team-ocean/veros">
    <img src="https://travis-ci.com/team-ocean/veros.svg?branch=master" alt="Build status">
  </a>
  <a href="https://codecov.io/gh/team-ocean/veros">
    <img src="https://codecov.io/gh/team-ocean/veros/branch/master/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://zenodo.org/badge/latestdoi/87419383">
    <img src="https://zenodo.org/badge/87419383.svg" alt="DOI">
  </a>
</p>

Veros, the *versatile ocean simulator*, is just that: A powerful tool
that makes high-performance ocean modeling approachable and fun. Since
it is a pure Python module, the days of struggling with complicated
model setup workflows, ancient programming environments, and obscure
legacy code are finally over.

Veros supports both a NumPy backend for small-scale problems and a
high-performance backend [powered by JAX](https://github.com/google/jax)
with CPU and GPU support, and runs on distributed architectures via MPI.

A good starting point to gain an overview of Veros\' design,
performance, and capabilities are [these slides of a talk on
Veros](http://slides.com/dionhaefner/veros-ams) held during the 98th
Annual Meeting of the American Meteorological Society.

The underlying numerics are based on
[pyOM2](https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2), an ocean model
developed by Carsten Eden (Institut für Meereskunde, Hamburg
University). Veros is currently being developed at Niels Bohr Institute,
Copenhagen University.

#### How about a demonstration?

<p align="center">
  <a href="https://vimeo.com/391237951">
      <img src="doc/_images/veros-preview.gif?raw=true" alt="0.25×0.25° high-resolution model spin-up">
  </a>
</p>

<p align="center">
(0.25×0.25° high-resolution model spin-up, click for better
quality)
</p>

## Features

Veros provides

-   a fully staggered **3-D grid geometry** (*C-grid*)
-   support for both **idealized and realistic configurations** in
    Cartesian or pseudo-spherical coordinates
-   several **friction and advection schemes** to choose from
-   isoneutral mixing, eddy-kinetic energy, turbulent kinetic energy,
    and internal wave energy **parameterizations**
-   several **pre-implemented diagnostics** such as energy fluxes,
    variable time averages, and a vertical overturning stream function
    (written to netCDF output)
-   **pre-configured idealized and realistic set-ups** that are ready to
    run and easy to adapt
-   **accessibility, readability, and extensibility** - thanks to the
    power of Python!

## Veros for the impatient

A minimal example to install and run Veros:

```bash
$ pip install veros
$ veros copy-setup acc --to /tmp
$ cd /tmp/acc
$ python acc.py
```

For more detailed installation instructions, have a look at [our
documentation](http://veros.readthedocs.io).

## Basic usage

To run Veros, you need to set up a model - i.e., specify which settings
and model domain you want to use. This is done by subclassing the
`Veros` base class in a *setup script* that is written in Python. You
should  use the `veros copy-setup` command to copy one into your current
folder. A good place to start is the
[ACC model](https://github.com/team-ocean/veros/blob/master/setup/acc/acc.py):

```bash
$ veros copy-setup acc
```

After setting up your model, all you need to do is call the `setup` and
`run` methods on your setup class. The pre-implemented setups can all be
executed as scripts, e.g. through

```bash
$ python acc.py
```

For more information on using Veros, have a look at [our
documentation](http://veros.readthedocs.io).

## Contributing

Contributions to Veros are always welcome, no matter if you spotted an
inaccuracy in [the documentation](http://veros.readthedocs.io), wrote a
nice setup, fixed a bug, or even extended Veros\' core mechanics. There
are two ways to contribute:

-   If you want to report a bug or request a missing feature, please
    [open an issue](https://github.com/team-ocean/veros/issues). If you
    are reporting a bug, make sure to include all relevant information
    for reproducing it (ideally through a *minimal* code sample).
-   If you want to fix the issue yourself, or wrote an extension for
    Veros - great! You are welcome to submit your code for review by
    committing it to a repository and opening a [pull
    request](https://github.com/team-ocean/veros/pulls). However,
    before you do so, please check [the contribution
    guide](http://veros.readthedocs.io/quickstart/get-started.html#enhancing-veros)
    for some tips on testing and benchmarking, and to make sure that
    your modifications adhere with our style policies. Most importantly,
    please ensure that you follow the [PEP8
    guidelines](https://www.python.org/dev/peps/pep-0008/), use
    *meaningful* variable names, and document your code using
    [Google-style
    docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
