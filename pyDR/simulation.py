"""
Functions for running a large number of simulations.

@author: Maximilian Balandat
@date Aug 13, 2016
"""

import os
import numpy as np
import pandas as pd
import logging
import logging.config
# logutils package required as QueueHandler/Listener not available <3.2
import logutils.queue

from pyDR.dynamic_models import FraukesModel, QuadraticUtilityWithBattery
from .blopt import BLModel, compute_BLtaking_eq
from .utils import (net_benefits_test, meter_charges, non_gen_tariffs,
                    pdp_compatible, get_energy_charges, create_folder)


def get_occupancy(index):
    """
        Function returning the occupancy of the building in the HVAC model
        based on the values used in:
        R. Gondhalekar, F. Oldewurtel, and C. N. Jones. Least-restrictive
        robust periodic model predictive control applied to room temperature
        regulation. Automatica, 49(9):2760 - 2766, 2013.
        We also add additional "occupancy" by appliances and devices in the
        non-work hours.
    """
    idx = index.tz_convert('US/Pacific')
    occ = (90*((idx.hour >= 8) & (idx.hour < 18)) +
           60*((idx.hour >= 18) & (idx.hour < 20)) +
           20*((idx.hour >= 20) & (idx.hour < 8)))
    return pd.DataFrame({'occupancy': occ}, index=idx).tz_convert('GMT')


def get_comfort_constraints(index):
    """
        Function returning the comfort constraints for the HVAC model.
    """
    idx = index.tz_convert('US/Pacific')
    isWorkHour = (idx.hour >= 8) & (idx.hour < 20)
    x1max = 26 * isWorkHour + 30 * ~isWorkHour
    x1min = 21 * isWorkHour + 19 * ~isWorkHour
    xmax = np.hstack([x1max[:, np.newaxis], np.ones((len(x1max), 2)) * np.NaN])
    xmin = np.hstack([x1min[:, np.newaxis], np.ones((len(x1max), 2)) * np.NaN])
    return xmax, xmin


def process_HVAC(blmodel, meter_per_day, ts_start, ts_end, tariff, LMP, node,
                 DR_type, n_DR, carbon_indiv=False, carbon_soc=False,
                 **kwargs):
    if kwargs.get('isRT'):
        tariff_name = tariff + '_RTP'
    elif kwargs.get('isPDP'):
        tariff_name = tariff + '_PDP'
    else:
        tariff_name = tariff
    meter_charge = meter_per_day*(ts_end - ts_start).days
    indiv_cost = blmodel._model.getObjective().getValue() + meter_charge
    energy_charge = np.sum([
            ec.getValue() for ec in blmodel.energy_charges(
                tariff, isRT=kwargs.get('isRT'), LMP=LMP,
                carbon=carbon_indiv, isPDP=kwargs.get('isPDP')).values])
    gen_cost = blmodel.generation_cost(LMP, carbon=carbon_soc).getValue()

    result_dict = {'node': node, 'tariff': tariff_name, 'DR_type': DR_type,
                   'n_DR': n_DR, 'energy_charge': energy_charge,
                   'meter_charge': meter_charge, 'indiv_cost': indiv_cost,
                   'gen_cost': gen_cost}
    if kwargs.get('output_folder') is not None:
        filename = '_'.join([tariff_name, node, str(ts_start.year),
                             DR_type, str(n_DR)])
        result_df = blmodel.get_results()
        result_df.name = filename
        filename = os.path.join(kwargs['output_folder'], filename) + '.pickle'
        result_df.to_pickle(filename)
        result_dict['output_file'] = filename
    return result_dict


def process_QU(blmodel, meter_per_day, ts_start, ts_end, tariff, LMP, node,
               DR_type, n_DR, eta, bat_size, carbon=False, **kwargs):
    meter_charge = meter_per_day*(ts_end - ts_start).days
    CS = -blmodel._model.getObjective().getValue() - meter_charge
    U = blmodel._dynsys.get_consumption_utilities(vals=True).values.sum()
    E = U - CS
    C = blmodel.generation_cost(LMP, carbon=carbon).getValue()
    RS = E - C
    SS = CS + RS
    if 'isRT' in kwargs:
        if kwargs['isRT']:
            tariff_name = tariff + '_RTP'
    elif 'isPDP' in kwargs:
        if kwargs['isPDP']:
            tariff_name = tariff + '_PDP'
    else:
        tariff_name = tariff
    result_dict = {'eta': eta, 'node': node, 'tariff': tariff_name,
                   'DR_type': DR_type, 'n_DR': n_DR, 'bat_size': bat_size,
                   'CS': CS, 'RS': RS, 'SS': SS, 'U': U, 'E': E, 'C': C}
    if kwargs.get('output_folder') is not None:
        filename = '_'.join([tariff_name, node, str(ts_start.year), str(eta),
                             DR_type, str(n_DR), bat_size])
        result_df = blmodel.get_results()
        result_df.name = filename
        filename = os.path.join(kwargs['output_folder'], filename) + '.pickle'
        result_df.to_pickle(filename)
        result_dict['output_file'] = filename
    return result_dict


def simulate_HVAC(i, log_queue, result_queue, data, nodes, tariffs, n_DR=[],
                  BLtaking=True, x0=[22, 22, 22], ignore_days=7, maxperday=2,
                  expMA=False, carbon=False, **kwargs):
    """
        Main function for simulating building model for a given dataset.
    """
    # in python > 3.4 use logging.handlers.QueueHandler instead
    qh = logutils.queue.QueueHandler(log_queue)
    logger = logging.getLogger('Process {}'.format(i))
    # start work
    logger.log(logging.INFO, 'Solving for range {} - {}'.format(
            data.index[0].date(), data.index[-1].date()))
    blmodel = BLModel('bdg_model')
    blmodel._model.setParam('LogToConsole', 0)  # supress gurobi cosole output
    GRB_log = kwargs.get('GRB_logfile')
    if GRB_log is not None:
        create_folder(GRB_log)
        blmodel._model.setParam('LogFile', GRB_log)
    else:
        blmodel._model.setParam('LogFile', '')
    if 'MIPGap' in kwargs:
        blmodel._model.setParam('MIPGap', kwargs['MIPGap'])
    blmodel.set_dynsys(FraukesModel(blmodel.get_model(), ts=60))
    index = data.index
    # create empty DataFrame to be filled with results
    results = pd.DataFrame()
    ts_start, ts_end = data.index[0], data.index[-1]
    for node in nodes:
        logger.log(logging.INFO, 'Solving for node {}'.format(node))
        # generate disturbance vector v
        v = data[[node+'_temp', node+'_solar', 'occupancy']].loc[index].values
        # set initial state and simulation horizon
        blmodel._dynsys.set_opts(x0=np.asarray(x0), T=len(index))
        blmodel.set_window(index)
        # get comfort constraints
        xmax, xmin = get_comfort_constraints(index)
        # define constraints and energy coefficients
        umin = np.array([0, 0])
        umax = np.array([500, kwargs['max_cool'][node]])
        blmodel._dynsys.set_opts(
            umin=np.tile(umin, (len(index), 1)),
            umax=np.tile(umax, (len(index), 1)),
            xmin=xmin, xmax=xmax)
        blmodel._dynsys.set_opts(nrg_coeffs=[1, 4])
        # populate model
        blmodel._dynsys.populate_model(v=v)
        # get LMPs
        LMP = data.loc[index][node+'_LMP']
        # loop through tariffs
        for tariff in tariffs:
            if 'E19' in tariff:
                meter_per_day = meter_charges[
                    tariff]['Voluntary']['SmartMeter']
            elif 'Zero' in tariff or 'OptFlat' in tariff:
                meter_per_day = 0.0
            else:
                meter_per_day = meter_charges[tariff]
            if tariff == 'Zero':
                logger.log(logging.INFO, 'Solving {} under tariff {}'.format(
                    node, tariff))
                blmodel.optimize(tariff, LMP=LMP, isRT=True, carbon=carbon)
                results = results.append(
                    process_HVAC(blmodel, meter_per_day, ts_start, ts_end,
                                 tariff, LMP, node, 'None', 0, isRT=True,
                                 carbon_indiv=carbon, carbon_soc=carbon,
                                 output_folder=kwargs.get('output_folder')),
                    ignore_index=True)
            else:
                if 'OptFlat' in tariff:
                    carbkw = carbon
                else:
                    carbkw = False
                # basic solution under the tariff
                logger.log(logging.INFO, 'Solving {} under tariff {}'.format(
                    node, tariff))
                # LMP here is necesssary for OptFlat tariff
                blmodel.optimize(tariff, LMP=LMP, carbon=carbkw)
                results = results.append(
                    process_HVAC(blmodel, meter_per_day, ts_start, ts_end,
                                 tariff, LMP, node, 'None', 0,
                                 carbon_indiv=carbkw, carbon_soc=carbon,
                                 output_folder=kwargs.get('output_folder')),
                    ignore_index=True)
                # solve for the RTP version of the tariff if applicable
                if tariff in non_gen_tariffs:
                    blmodel.optimize(tariff, LMP=LMP, isRT=True, carbon=carbkw)
                    results = results.append(
                        process_HVAC(blmodel, meter_per_day, ts_start, ts_end,
                                     tariff, LMP, node, 'None', 0, isRT=True,
                                     carbon_indiv=carbkw, carbon_soc=carbon,
                                     output_folder=kwargs.get(
                                        'output_folder')),
                        ignore_index=True)
                # solve for the PDP version of the tariff if applicable
                if tariff in pdp_compatible:
                    blmodel.optimize(tariff, LMP=LMP, isPDP=True,
                                     carbon=carbkw)
                    results = results.append(
                        process_HVAC(blmodel, meter_per_day, ts_start, ts_end,
                                     tariff, LMP, node, 'None', 0, isPDP=True,
                                     carbon_indiv=carbkw, carbon_soc=carbon,
                                     output_folder=kwargs.get(
                                        'output_folder')),
                        ignore_index=True)
                # now deal with DR events
                for ndr in n_DR:
                    isDR = net_benefits_test(LMP, n=ndr, how='absolute',
                                             ignore_days=ignore_days,
                                             maxperday=maxperday)
                    dr_periods = isDR.index[isDR]
                    # CAISO
                    logger.log(logging.INFO,
                               'Solving {} / {} '.format(node, tariff) +
                               'for {} events under CAISO'.format(ndr))
                    blmodel.optimize(tariff, LMP=LMP, dr_periods=dr_periods,
                                     BL='CAISO', carbon=carbkw)
                    results = results.append(
                        process_HVAC(blmodel, meter_per_day, ts_start, ts_end,
                                     tariff, LMP, node, 'CAISO', ndr,
                                     carbon_indiv=carbkw, carbon_soc=carbon,
                                     output_folder=kwargs.get(
                                        'output_folder')),
                        ignore_index=True)
                    # CAISO for LMP-G (if applicable)
                    # ToDo: Warm-start using solution from LMP compensation
                    if tariff in non_gen_tariffs:
                        logger.log(
                            logging.INFO,
                            'Solving {} / {} for '.format(node, tariff) +
                            ' {} events under CAISO (LMP-G)'.format(ndr))
                        blmodel.optimize(tariff, LMP=LMP,
                                         dr_periods=dr_periods, BL='CAISO',
                                         isLMPmG=True, carbon=carbkw)
                        results = results.append(
                            process_HVAC(
                                blmodel, meter_per_day, ts_start, ts_end,
                                tariff, LMP, node, 'CAISO_LMP-G', ndr,
                                carbon_indiv=carbkw, carbon_soc=carbon,
                                output_folder=kwargs.get('output_folder')),
                            ignore_index=True)
                    # expMA
                    if expMA:
                        # ToDo: Warm-start using solution from CAISO BL
                        logger.log(
                            logging.INFO,
                            'Solving {} / {} for '.format(node, tariff) +
                            ' {} events under expMA'.format(ndr))
                        blmodel.optimize(
                            tariff, LMP=LMP, dr_periods=dr_periods, BL='expMA',
                            alpha_b=0.175, alpha_nb=0.25, carbon=carbkw)
                        results = results.append(
                            process_HVAC(blmodel, meter_per_day, ts_start,
                                         ts_end, tariff, LMP, node, 'expMA',
                                         ndr, carbon_indiv=carbkw,
                                         carbon_soc=carbon,
                                         output_folder=kwargs.get(
                                            'output_folder')),
                            ignore_index=True)
                        # expMA for LMP-G (if applicable)
                        if tariff in non_gen_tariffs:
                            logger.log(
                                logging.INFO,
                                'Solving {} / {} for '.format(node, tariff) +
                                ' {} events under expMA (LMP-G)'.format(ndr))
                            blmodel.optimize(
                                tariff, LMP=LMP, dr_periods=dr_periods,
                                BL='expMA', alpha_b=0.175, alpha_nb=0.25,
                                isLMPmG=True, carbon=carbkw)
                            results = results.append(
                                process_HVAC(blmodel, meter_per_day, ts_start,
                                             ts_end, tariff, LMP, node,
                                             'expMA_LMP-G', ndr,
                                             carbon_indiv=carbkw,
                                             carbon_soc=carbon,
                                             output_folder=kwargs.get(
                                                'output_folder')),
                                ignore_index=True)
                    # compute BL-taking equilibrium
                    if BLtaking:
                        # CAISO BL
                        compute_BLtaking_eq(
                            blmodel, tariff, LMP, dr_periods, BL='CAISO',
                            blinit='noDR', eps=0.01, maxiter=10, logger=logger,
                            carbon=carbkw, **kwargs)
                        results = results.append(
                            process_HVAC(blmodel, meter_per_day, ts_start,
                                         ts_end, tariff, LMP, node,
                                         'CAISO_BLT', ndr, carbon_indiv=carbkw,
                                         carbon_soc=carbon,
                                         output_folder=kwargs.get(
                                            'output_folder')),
                            ignore_index=True)
                        # do the same thing for LMP-G if applicable
                        # ToDo: Warm-start using solution from LMP compensation
                        if tariff in non_gen_tariffs:
                            compute_BLtaking_eq(
                                blmodel, tariff, LMP, dr_periods, BL='CAISO',
                                blinit='noDR', eps=0.01, maxiter=10,
                                logger=logger, isLMPmG=True, carbon=carbkw,
                                **kwargs)
                            results = results.append(
                                process_HVAC(blmodel, meter_per_day, ts_start,
                                             ts_end, tariff, LMP, node,
                                             'CAISO_LMP-G_BLT', ndr,
                                             carbon_indiv=carbkw,
                                             carbon_soc=carbon,
                                             output_folder=kwargs.get(
                                                'output_folder')),
                                ignore_index=True)
                        # expMA BL
                        if expMA:
                            # ToDo: Warm-start using solution from CAISO BL
                            compute_BLtaking_eq(
                                blmodel, tariff, LMP, dr_periods, BL='expMA',
                                blinit='noDR', eps=0.01, maxiter=10,
                                logger=logger, carbon=carbkw, **kwargs)
                            results = results.append(
                                process_HVAC(blmodel, meter_per_day, ts_start,
                                             ts_end, tariff, LMP, node,
                                             'expMA_BLT', ndr,
                                             carbon_indiv=carbkw,
                                             carbon_soc=carbon,
                                             output_folder=kwargs.get(
                                                'output_folder')),
                                ignore_index=True)
                            # do the same thing for LMP-G if applicable
                            if tariff in non_gen_tariffs:
                                compute_BLtaking_eq(
                                    blmodel, tariff, LMP, dr_periods,
                                    BL='expMA', blinit='noDR', eps=0.01,
                                    maxiter=10, logger=logger, isLMPmG=True,
                                    carbon=carbkw, **kwargs)
                                results = results.append(
                                    process_HVAC(
                                        blmodel, meter_per_day, ts_start,
                                        ts_end, tariff, LMP, node,
                                        'expMA_LMP-G_BLT', ndr,
                                        carbon_indiv=carbkw, carbon_soc=carbon,
                                        output_folder=kwargs.get(
                                            'output_folder')),
                                    ignore_index=True)
    results['start_date'] = ts_start.date()
    results['end_date'] = ts_end.date()
    result_queue.put(results)
    logger.log(logging.INFO, 'Simulation completed successfully.')
    return


def simulate_QU(i, log_queue, result_queue, data, etas, nodes, tariffs, xlims,
                ulims, x0=0, n_DR=[], BLtaking=True, ignore_days=7,
                maxperday=2, carbon=False, **kwargs):
    """
        Main function for simulating building model for a given dataset.
    """
    # in python > 3.4 use logging.handlers.QueueHandler instead
    qh = logutils.queue.QueueHandler(log_queue)
    logger = logging.getLogger('Process {}'.format(i))
    # start work
    logger.log(logging.INFO, 'Solving for range {} - {}'.format(
            data.index[0].date(), data.index[-1].date()))
    blmodel = BLModel('QuadUtilModel')
    blmodel._model.setParam('LogToConsole', 0)  # supress gurobi cosole output
    GRB_log = kwargs.get('GRB_logfile')
    if GRB_log is not None:
        create_folder(GRB_log)
        blmodel._model.setParam('LogFile', GRB_log)
    else:
        blmodel._model.setParam('LogFile', '')
    if 'MIPGap' in kwargs:
        blmodel._model.setParam('MIPGap', kwargs['MIPGap'])
    if 'TimeLimit' in kwargs:
        blmodel._model.setParam('TimeLimit', kwargs['TimeLimit'])
    # create dynamic model
    index = data.index
    qu_mdl = QuadraticUtilityWithBattery(blmodel.get_model(), ts=60)
    # need to set this before we can calibrate the model later
    blmodel.set_dynsys(qu_mdl)
    blmodel.set_window(index)
    # set some options
    qu_mdl.set_opts(x0=np.array([x0]))
    qu_mdl.set_opts(T=len(index))
    qu_mdl.set_opts(nrg_coeffs=[1, 0, 1])
    # define DataFrame to be filled with results
    results = pd.DataFrame()
    ts_start, ts_end = data.index[0], data.index[-1]
    for eta in etas:
        for tariff in tariffs:
            # get basic tariff info (for calibration / constraints)
            if 'Zero' in tariff:
                base_tar = tariff.strip('Zero')
            elif 'OptFlat' in tariff:
                base_tar = tariff.strip('OptFlat')
            else:
                base_tar = tariff
            # deal with meter charges
            if 'E19' in tariff:
                meter_per_day = meter_charges[
                    tariff]['Voluntary']['SmartMeter']
            elif 'Zero' in tariff or 'OptFlat' in tariff:
                meter_per_day = 0.0
            else:
                meter_per_day = meter_charges[tariff]
            for bat_size in xlims.keys():
                logger.log(logging.INFO, 'Calibrating utility for tariff ' +
                           ' {} under eta={} with bat_size={}'.format(
                            tariff, eta, bat_size))
                qu_mdl.set_opts(
                    xmin=np.tile(xlims[bat_size][base_tar][0],
                                 (len(index), 1)),
                    xmax=np.tile(xlims[bat_size][base_tar][1],
                                 (len(index), 1)),
                    umin=np.tile([ulims[bat_size][base_tar][0][0],
                                  ulims[bat_size][base_tar][1][0],
                                  ulims[bat_size][base_tar][2][0]],
                                 (len(index), 1)),
                    umax=np.tile([ulims[bat_size][base_tar][0][1],
                                  ulims[bat_size][base_tar][1][1],
                                  ulims[bat_size][base_tar][2][1]],
                                 (len(index), 1)))
                # Compute utility parameters
                qu_mdl.compute_util_params(
                    kwargs['loadshapes'][kwargs['load_map'][base_tar]],
                    get_energy_charges(
                        index, kwargs['charge_map'][base_tar])['EnergyCharge'],
                    eta=eta)
                # choose energy coefficients and populate model
                qu_mdl.populate_model()
                for node in nodes:
                    # get LMPs
                    LMP = data.loc[index][node+'_LMP']
                    if 'Zero' in tariff:
                        logger.log(logging.INFO, 'Solving {} '.format(node) +
                                   ' under tariff {}'.format(tariff))
                        blmodel.optimize(tariff, LMP=LMP, isRT=True,
                                         carbon=carbon)
                        results = results.append(
                            process_QU(blmodel, meter_per_day, ts_start,
                                       ts_end, tariff, LMP, node, 'None', 0,
                                       eta, bat_size, isRT=True, carbon=carbon,
                                       output_folder=kwargs.get(
                                        'output_folder')),
                            ignore_index=True)
                    else:
                        if 'OptFlat' in tariff:
                            carbkw = carbon
                        else:
                            carbkw = False
                        # vanilla solution under the tariff
                        logger.log(
                            logging.INFO, 'Solving {} under tariff {}'.format(
                                node, tariff))
                        blmodel.optimize(tariff, LMP=LMP, carbon=carbkw)
                        results = results.append(
                            process_QU(blmodel, meter_per_day, ts_start,
                                       ts_end, tariff, LMP, node, 'None', 0,
                                       eta, bat_size, carbon=carbon,
                                       output_folder=kwargs.get(
                                        'output_folder')),
                            ignore_index=True)
                        # solve for the RTP version of the tariff if applicable
                        if (tariff in non_gen_tariffs and
                                'OptFlat' not in tariff):
                            blmodel.optimize(tariff, LMP=LMP, isRT=True,
                                             carbon=carbkw)
                            results = results.append(
                                process_QU(blmodel, meter_per_day, ts_start,
                                           ts_end, tariff, LMP, node, 'None',
                                           0, eta, bat_size, carbon=carbon,
                                           isRT=True, output_folder=kwargs.get(
                                            'output_folder')),
                                ignore_index=True)
                        # solve for the PDP version of the tariff if applicable
                        if tariff in pdp_compatible:
                            blmodel.optimize(tariff, LMP=LMP, isPDP=True,
                                             carbon=carbkw)
                            results = results.append(
                                process_QU(blmodel, meter_per_day, ts_start,
                                           ts_end, tariff, LMP, node, 'None',
                                           0, eta, bat_size, carbon=carbon,
                                           isPDP=True,
                                           output_folder=kwargs.get(
                                            'output_folder')),
                                ignore_index=True)
                        # now deal with DR events
                        for ndr in n_DR:
                            isDR = net_benefits_test(
                                LMP, n=ndr, how='absolute',
                                ignore_days=ignore_days, maxperday=maxperday)
                            dr_periods = isDR.index[isDR]
                            # CAISO
                            logger.log(
                                logging.INFO,
                                'Solving {} / {} '.format(node, tariff) +
                                'for {} events under CAISO'.format(ndr))
                            blmodel.optimize(
                                tariff, LMP=LMP, dr_periods=dr_periods,
                                BL='CAISO', carbon=carbkw)
                            results = results.append(
                               process_QU(blmodel, meter_per_day, ts_start,
                                          ts_end, tariff, LMP, node, 'CAISO',
                                          ndr, eta, bat_size, carbon=carbon,
                                          output_folder=kwargs.get(
                                            'output_folder')),
                               ignore_index=True)
                            # CAISO for LMP-G (if applicable)
                            if tariff in non_gen_tariffs:
                                logger.log(
                                    logging.INFO,
                                    'Solving {} / {} '.format(node, tariff) +
                                    'for {} events under CAISO (LMP-G)'.format(
                                        ndr))
                                blmodel.optimize(
                                    tariff, LMP=LMP, dr_periods=dr_periods,
                                    BL='CAISO', isLMPmG=True, carbon=carbkw)
                                results = results.append(
                                    process_QU(
                                        blmodel, meter_per_day, ts_start,
                                        ts_end, tariff, LMP, node,
                                        'CAISO_LMP-G', ndr, eta, bat_size,
                                        carbon=carbon,
                                        output_folder=kwargs.get(
                                            'output_folder')),
                                    ignore_index=True)
                            # compute BL-taking equilibrium
                            if BLtaking:
                                # CAISO BL
                                compute_BLtaking_eq(
                                    blmodel, tariff, LMP, dr_periods,
                                    BL='CAISO', blinit='noDR', eps=0.01,
                                    maxiter=10, logger=logger, carbon=carbkw,
                                    **kwargs)
                                results = results.append(
                                   process_QU(blmodel, meter_per_day, ts_start,
                                              ts_end, tariff, LMP, node,
                                              'CAISO_BLT', ndr, eta, bat_size,
                                              carbon=carbon,
                                              output_folder=kwargs.get(
                                                'output_folder')),
                                   ignore_index=True)
                                # do the same thing for LMP-G if applicable
                                if tariff in non_gen_tariffs:
                                    compute_BLtaking_eq(
                                        blmodel, tariff, LMP, dr_periods,
                                        BL='CAISO', blinit='noDR', eps=0.01,
                                        maxiter=10, logger=logger,
                                        isLMPmG=True, carbon=carbkw, **kwargs)
                                    results = results.append(
                                        process_QU(
                                            blmodel, meter_per_day, ts_start,
                                            ts_end, tariff, LMP, node,
                                            'CAISO_LMP-G_BLT', ndr, eta,
                                            bat_size, carbon=carbon,
                                            output_folder=kwargs.get(
                                                'output_folder')),
                                        ignore_index=True)
    results['start_date'] = ts_start.date()
    results['end_date'] = ts_end.date()
    result_queue.put(results)
    logger.log(logging.INFO, 'Simulation completed successfully.')
    return


def log_config(logfile):
    """
        Return dictionary for configuring the root logger
    """
    cfg = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': ('%(asctime)s %(name)-10s %(levelname)-6s' +
                           '%(processName)-10s: %(message)s')
            },
            'simple': {
                'class': 'logging.Formatter',
                'format': ('%(name)-10s %(levelname)-6s ' +
                           ' %(processName)-10s: %(message)s')
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': os.path.expanduser(logfile),
                'mode': 'a',
                'formatter': 'detailed',
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
    return cfg


# Define the maximum cooling power (in kW) of the HVAC system
max_cool = {'PGEB': 150, 'PGP2': 150, 'PGCC': 200, 'PGSA': 300,
            'SCEW': 250, 'PGF1': 300}
