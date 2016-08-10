# pyDR beta 0.1 - August 2016
Authors: Maximilian Balandat, Clay Campaigne, Lillian Ratliff

Python package for simulating behavior and quantifying welfare effects of electricity consumers facing various dynamic pricing schemes.

### Requirements / Dependencies: 
* gurobipy (+ GUROBI with valid license)
* numpy & scipy
* pandas
* pytz
* patsy

### Modules:
* dynamic_models.py: Definition of basic dynamic consumption models
* blopt.py: Main module defining model and formulating optimization problem
* utils.py: Specify tariff data and provide utility functions for other modules

### Installation:
Simply add the folder pyDR to the python path 

### Usage:
See scripts in examples folder (work in progress)

### Data:
To run the scripts in the examples folder you will need to download and extract the following [csv file](https://www.ocf.berkeley.edu/~balandat/data_complete_withsolar_corrected.csv.zip) (~24MB)
