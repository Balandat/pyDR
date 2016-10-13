"""
Run a large number of simulations for the Quadratic Uitlity model.
Results from these simulations are reported in:
C. Campaigne, M. Balandat and L. Ratliff: Welfare Effects
of Dynamic Electricity Pricing. In preparation.

@author: Maximilian Balandat
@date Aug 13, 2016
"""

# import packages and set up things
import multiprocessing as mp
import pandas as pd
import logging
# logutils package required as QueueHandler/Listener not available <3.2
import logutils.queue
import logging.config
from datetime import datetime
from pyDR.simulation import log_config, simulate_QU


############################################################################
# Setup

# location of data files (available for download at
# https://www.ocf.berkeley.edu/~balandat/pyDR_data.zip)
data_file = 'PATH_TO_DATA/data_complete.csv'
loadshape_file = 'PATH_TO_DATA/loadshapes.csv'

# location of the log file
log_file = 'PATH_TO_LOG/QU_sim.log'

# directory for GUROBI log files
GRB_logdir = 'PATH_TO_LOG/GRB_logs/'

# location of the result file
result_file = 'PATH_TO_RESULTS/results.csv'

# folder for output files (Attention: If not none then this will
# save a few GB of .pickle files)
output_folder = None

############################################################################

# read in data and PG&E loadshapes (download these files from
# )
data = pd.read_csv(data_file, parse_dates=['timestamp_GMT'],
                   index_col='timestamp_GMT').tz_localize('GMT')
data = data.resample('1H').mean()
loadshapes = pd.read_csv(loadshape_file, parse_dates=['timestamp_GMT'],
                         index_col='timestamp_GMT').tz_localize('GMT')
load_map = {'A1': 'A1', 'A1TOU': 'A1', 'A6TOU': 'A1'}
charge_map = {'A1': 'A1', 'A1TOU': 'A1', 'A6TOU': 'A1'}


# Define model and simulation parameters

# generate copies of input data for parallelization
sim_ranges = [[datetime(2012, 1, 1), datetime(2012, 12, 31)],
              [datetime(2013, 1, 1), datetime(2013, 12, 31)],
              [datetime(2014, 1, 1), datetime(2014, 12, 31)]]
sim_tariffs = ['OptFlatA1']
sim_nodes = ['PGCC', 'PGEB', 'PGF1', 'PGP2', 'PGSA']
n_DR = [25, 50, 75]
n_ranges = len(sim_ranges)

etas = [-0.05, -0.1, -0.2, -0.3]

# battery charge limits (lower/upper) in kWh
xlims = {'Medium': {'A1':         [0, 10],
                    'A1TOU':      [0, 10],
                    'A6TOU':      [0, 10]},
         'Large':  {'A1':         [0, 25],
                    'A1TOU':      [0, 25],
                    'A6TOU':      [0, 25]}}
# limits on charging, discharging, direct consumption [lower, upper]
ulims = {'Medium': {'A1':         [[0,  5], [0, 7.5], [0,  50]],
                    'A1TOU':      [[0,  5], [0, 7.5], [0,  50]],
                    'A6TOU':      [[0, 5], [0,  7.5], [0, 50]]},
         'Large':  {'A1':         [[0, 25], [0,  30], [0,  100]],
                    'A1TOU':      [[0, 25], [0,  30], [0,  100]],
                    'A6TOU':      [[0, 25], [0,  30], [0, 100]]}}

# generate scaled sub-DataFrame
data_scaled = pd.concat(
    [data[[node+'_LMP']] for node in sim_nodes],
    axis=1)

# generate a list of DataFrames of different ranges for parallelization
data_par = []
for (start_date, end_date) in sim_ranges:
    ts_start = pd.Timestamp(start_date, tz='US/Pacific')
    ts_end = pd.Timestamp(end_date, tz='US/Pacific')
    data_par.append(data_scaled[(data_scaled.index >= ts_start) &
                                (data_scaled.index <= ts_end)])

# configure logger
logging.config.dictConfig(log_config(log_file))
log_queue = mp.Queue(-1)
root = logging.getLogger()
ql = logutils.queue.QueueListener(log_queue, *root.handlers)

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
            target=simulate_QU, name='sim_worker {}'.format(i),
            args=(i, log_queue, result_queue, data_par[i], etas,
                  sim_nodes, sim_tariffs, xlims, ulims),
            kwargs={'n_DR': n_DR, 'BLtaking': True, 'carbon': True,
                    'loadshapes': loadshapes, 'load_map': load_map,
                    'charge_map': charge_map, 'output_folder': output_folder,
                    'GRB_logfile': GRB_logdir + 'GRB_{}.log'.format(i),
                    'MIPGap': 5e-5, 'TimeLimit': 1500})
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
