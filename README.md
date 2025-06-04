# PACSim Light-Control Beta Version

## Overview
This repository gives a Beta version of the PACSim software, which is a framework for simulation of ionic colloidal assembly using MD simulations in OpenMM. The framework was primarily authored by Philipp Hoellmer with contributions from others in the Hocky group, but this particular version contains special modifications from Nicole Smina for the following publication:

Light-Controlled Crystallization

Steven van Kesteren, Nicole Smina, Shihao Zang, Cheuk Wai Leung, Glen Hocky and Stefano Sacannaa

Example inputs and analysis scripts for this paper can be found at [a GitHub page for that paper](https://github.com/hocky-research-group/light_controlled_crystallization)

# Installation instructions

## Dependencies

If you only want to use the openmm part of this package, you can use Python 3.12 (or any older version >= 3.10).

If you want to use the Hoomd part of this package, use Python 3.10 because that is the latest Python version that is 
supported by Hoomd < 3.

We recommend installing Python and the required packages using 
[Anaconda](https://www.anaconda.com/products/distribution). 

The following packages are required:

- jupyterlab >= 4.1 
- matplotlib >= 3.8
- numpy >= 1.26
- openmm >= 8.0
- pytest >= 7.4
- gsd >= 3.2
- pyyaml >= 6.0
- tqdm >= 4.65
- pandas >= 2.2
- hoomd == 2.9.7

## Installation

Clone the repository and install the package in editable mode in your virtual environment using pip:

```bash
pip install -e .
```

Note that this attempts to install the requirements with pip, if you did not install them yourself before. However, 
because hoomd is not available on PyPI, you need to install it manually (or via conda).

## Testing

After installation, you can test whether your installation is working correctly by running the following command from 
this directory:

```bash
pytest colloids
```

If hoomd is not installed, some tests are automatically skipped.

## Usage

The installation process creates three executables `colloids-run`, `colloids-resume`, and `colloids-create`. You might 
have to add the directory where pip installs executables to your PATH environment variable in order to access these 
executables.

### colloids-run

The `colloids-run` executable is used to run simulations. It expects a configuration file in yaml format as the only 
positional argument:

```bash
colloids-run run.yaml
```

An exemplary configuration file called `example.yaml` can be created with the command 
`colloids-run --example`. Another exemplary configuration file is provided in [`colloids/run.yaml`](colloids/run.yaml).

### colloids-resume
A simulation that is run with the `colloids-run` executable creates checkpoints in periodic intervals. One can resume a 
simulation from a checkpoint using the `colloids-resume` executable. It expects the original configuration file (because
the checkpoint file only stores the positions and velocities of the particles in an OpenMM context), the checkpoint 
file, and the number of time steps that should be run (the corresponding value in the configuration file is ignored). 
For example, use the following command to continue a simulation for 100000 time steps:

```bash
colloids-resume run.yaml checkpoint.chk 100000
```

### colloids-create
The configuration file for the `colloids-run` executable specifies the filename of an initial configuration for the 
simulation in the `initial_configuration` key. This initial configuration should be stored in the [extended XYZ file 
format](https://www.ovito.org/manual/reference/file_formats/input/xyz.html).

The `colloids-create` executable can be used to create an initial configuration for simulations in the extended XYZ file 
format. It expects two configuration files in yaml format as positional arguments:
1.  A configuration file that specifies the parameters of the simulation. This yaml file is usually the one that is 
    passed to the `colloids-run` executable afterward, and it contains the filename in which the generated initial 
    configuration is stored. See [`colloids/run.yaml`](colloids/run.yaml) for an example.
2. A configuration file that specifies the parameters of the initial configuration. See 
   [`colloids/colloids_create/configuration.yaml`](colloids/colloids_create/configuration.yaml) for an example. Another 
   exemplary configuration file called `example_configuration.yaml` can be created with the command
   `colloids-create --example`.

A typical workflow for running a simulation with `colloids-run` from an initial configuration created by 
`colloids-create` consists of creating a directory with `run.yaml` and `configuration.yaml` files, and then running the 
following two commands:

```bash
colloids-create run.yaml configuration.yaml
colloids-run run.yaml
```

### colloids-analyze

The `colloids-run` executable generates a trajectory in the GSD file format that can be visualized with 
[Ovito](https://www.ovito.org) and 
analyzed with the [GSD](https://gsd.readthedocs.io/en/stable/python-api.html) Python package.

In addition, the `colloids-run` executable generates a CSV file that contains the time series of the potential energy,
the kinetic energy, and the temperature of the system. The `colloids-analyze` executable can be used to plot these time
series. Here, it can plot the results of several simulations at once.

The `colloids-analyze` expects a configuration file in yaml format that specifies the parameters of the analysis (like 
the output directory where the plots should be generated) as the first positional argument. An exemplary configuration 
file called `example_analysis.yaml` can be created with the command `colloids-analyze --example`. Another exemplary 
configuration file is provided in [`colloids/colloids_analyze/analysis.yaml`](colloids/colloids_analyze/analysis.yaml).

After this, the `colloids-analyze` executable receives an arbitrary number of configuration files that specified the 
parameters of the simulations that should be analyzed. These configuration files contain the name of the CSV files
that will be plotted.

Assume, for example, that you ran three simulations with `colloids-run` in the directories `Run1`, `Run2`, and `Run3` 
based on configuration files called `run.yaml` in either of these directories. You can analyze and compare the results 
of these simulations with the command:
```bash
colloids-analyze analysis.yaml Run1/run.yaml Run2/run.yaml Run3/run.yaml
```
