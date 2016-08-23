# pyDR beta 0.1 - August 2016
Authors: Maximilian Balandat, Clay Campaigne, Lillian Ratliff

Python package for simulating behavior and quantifying welfare effects of electricity consumers facing various dynamic pricing schemes.
This package is the basis for the results reported in:

*C. Campaigne, M. Balandat and L. Ratliff: Welfare Effects of Dynamic Electricity Pricing. In preparation.*

### Requirements / Dependencies:
* gurobipy (+ GUROBI with valid license)
* numpy
* scipy
* pandas
* pytz
* patsy
* logutils (for examples if using python 2)

### Modules:
* dynamic_models.py: Definition of basic dynamic consumption models
* blopt.py: Main module defining model and formulating optimization problem
* utils.py: Specify tariff data and provide utility functions for other modules
* simulation.py: Functions for simulating a large number of different scenarios

### Installation (using setuptools):
* from source: `python setup.py install`
* from pypi: `pip install pyDR`

### Usage:
See scripts in examples folder

### Data:
To run the scripts in the examples folder you will need to download and extract the following [zip file](https://www.ocf.berkeley.edu/~balandat/pyDR_data.zip) (~25MB)
