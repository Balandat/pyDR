"""
Run a large number of simulations for the HVAC model.
Results from these simulations are reported in:
C. Campaigne, M. Balandat and L. Ratliff: Welfare Effects
of Dynamic Electricity Pricing. In preparation.

@author: Maximilian Balandat
@date Sep 23, 2017
"""

# import packages and set up things
import os
import multiprocessing as mp
import pandas as pd
import logging
import logging.handlers
import logging.config
from datetime import datetime
from pyDR.simulation import get_internal_gains, log_config, simulate_HVAC, max_cool

############################################################################
# Setup

DATA_PATH = "PATH_TO_PYDR_DATA"
LOG_PATH = "PATH_TO_LOGS"
RESULTS_PATH = "PATH_TO_RESULTS"

# location of data files (available for download at
# https://www.ocf.berkeley.edu/~balandat/pyDR_data.zip)
data_file = os.path.join(DATA_PATH, "data_complete.csv")

# location of the log file
log_file = os.path.join(LOG_PATH, "HVAC_sim.log")

# directory for GUROBI log files
GRB_logdir = os.path.join(LOG_PATH, "GRB_logs")

# location of the result file
result_file = os.path.join(RESULTS_PATH, "results.csv")

# folder for output files (Attention: If not none then this will
# save a few GB of .pickle files)
output_folder = None

############################################################################

# read in data
data = pd.read_csv(data_file, parse_dates=['timestamp_GMT'],
                   index_col='timestamp_GMT').tz_convert('GMT')
data = data.resample('1H').mean()

# Define model and simulation parameters

# generate copies of input data for parallelization
sim_ranges = [[datetime(2012, 1, 1), datetime(2012, 12, 31)],
              [datetime(2013, 1, 1), datetime(2013, 12, 31)],
              [datetime(2014, 1, 1), datetime(2014, 12, 31)]]
sim_tariffs = ['Zero', 'OptFlat', 'A1', 'A1TOU', 'A6TOU', 'A10_secondary',
               'A10TOU_secondary', 'E19TOU_secondary']
sim_nodes = ['PGCC', 'PGEB', 'PGF1', 'PGP2', 'PGSA']
n_DR = [75]
n_ranges = len(sim_ranges)

# generate scaled sub-DataFrame
data_sim = pd.concat(
    [data[[node+'_temp']] for node in sim_nodes] +
    [data[[node+'_solar']] for node in sim_nodes] +
    [data[[node+'_LMP']] for node in sim_nodes] +
    [get_internal_gains(data.index)], axis=1)
# generate a list of DataFrames of different ranges for parallelization
data_par = []
for (start_date, end_date) in sim_ranges:
    ts_start = pd.Timestamp(start_date, tz='US/Pacific')
    ts_end = pd.Timestamp(end_date, tz='US/Pacific')
    data_par.append(
        data_sim[(data_sim.index >= ts_start) & (data_sim.index <= ts_end)]
    )

# configure logger
logging.config.dictConfig(log_config(log_file))
log_queue = mp.Queue(-1)
root = logging.getLogger()
ql = logging.handlers.QueueListener(log_queue, *root.handlers)

# start root logging via queue listener
ql.start()
root.log(logging.INFO, 'Starting simulation.')

results = []

# start simulating
with mp.Manager() as mngr:
    result_queue = mngr.Queue(-1)
    sim_workers = []
    for i in range(n_ranges):
        sim_worker = mp.Process(
            target=simulate_HVAC, name='sim_worker {}'.format(i),
            args=(i, log_queue, result_queue, data_par[i],
                  sim_nodes, sim_tariffs, n_DR),
            kwargs={'GRB_logfile': GRB_logdir + 'GRB_{}.log'.format(i),
                    'expMA': False, 'carbon': True, 'MIPGap': 1e-6,
                    'TimeLimit': 2000, 'output_folder': output_folder,
                    'max_cool': max_cool})
        sim_workers.append(sim_worker)
        sim_worker.start()

    # wait for all worker processes to finish
    for sw in sim_workers:
        sw.join()

    root.log(logging.DEBUG, 'Extracting results.')
    # extract results
    for i in range(n_ranges):
        results.append(result_queue.get())

# save results
root.log(logging.DEBUG, 'Saving results to disk.')
results = pd.concat(results, ignore_index=True)
results.to_csv(result_file, index=False)

# stop logging
root.log(logging.INFO, 'Simulation completed.')
ql.stop()
