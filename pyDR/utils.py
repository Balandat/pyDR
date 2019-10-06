"""
Same basic parameters for the Baselining work.

@author: Maximilian Balandat, Lillian Ratliff
@date Aug 30, 2016
"""

import numpy as np
import pandas as pd
import os

from itertools import chain, combinations
from scipy.signal import cont2discrete
from datetime import datetime
from pytz import timezone
from pandas.tseries.holiday import USFederalHolidayCalendar

E19 = ['E19TOU_secondary', 'E19TOU_primary', 'E19TOU_transmission']
# define social cost of carbon
# if cost per metric ton is $40:
carbon_costs = {2012: 16.60, 2013: 11.62, 2014: 11.62}
# # if cost per metric ton is $38:
# carbon_costs = {'2012': 15.77, '2013': 10.79, '2014': 10.79}


def create_folder(filename):
    """
        Helper function for safely creating all sub-directories
        for a specific file in python2
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def matrices_frauke(ts=15):
    """
        Return matrices A, B, and E of the discrete-time dynamical system model
        of Frauke's builing model with sampling time ts minutes.
    """
    # define matrices for Frauke's Building model
    # note: both input variables are assumed to be non-negative!
    c1, c2, c3 = 9.356e5, 2.97e6, 6.695e5
    k1, k2, k3, k4, k5 = 16.48, 108.5, 5.0, 30.5, 23.04
    Act = np.array([[-(k1+k2+k3+k5)/c1, (k1+k2)/c1, k5/c1],
                   [(k1+k2)/c2, -(k1+k2)/c2, 0],
                   [k5/c3, 0, -(k4+k5)/c3]])
    Bct = np.array([[1/c1, -1/c1],
                   [0, -0],
                   [0, -0]])
    Ect = np.array([[k3/c1, 1/c1, 1/c1],
                   [0, 1/c2, 0],
                   [k4/c3, 0, 0]])
    Cct = np.array([[0, 0, 0]])
    Dct = np.array([[0, 0]])
    # convert cont time matrices to discrete time using zoh
    (A, B, C, D, dt) = cont2discrete((Act, Bct, Cct, Dct), ts*60, method='zoh')
    (A, E, C, D, dt) = cont2discrete((Act, Ect, Cct, Dct), ts*60, method='zoh')
    return A, B, E

def matrices_pavlak(ts=15):
    """
        Return matrices A, B, and E of the discrete-time dynamical system model
        of Pavlak's builing model with sampling time ts minutes.
    """
    # define matrices for Pavlak's Building model
    # note: both input variables are assumed to be non-negative!
    # R's in K/kW and C's in kJ/K
    R1, R2 = 0.519956063919649, 0.00195669419889427
    R3, Rw = 0.00544245602566602, 0.156536600054259
    Cz, Cm = 215552.466637916, 8533970.42635405

    den = R2*R3 + Rw*R2 + Rw*R3

    Act = np.array([[Rw*R2/(R3*Cz*den) - 1/(R3*Cz), Rw/(Cz*den)],
                    [Rw/(Cm*den), -1/(R1*Cm)-1/(R2*Cm)+Rw*R3/(R2*Cm*den)]])
    Bct = np.array([[1/Cz, -1/Cz],
                    [0, -0]])
    Ect = np.array([[R2/(Cz*den), Rw*R2/(Cz*den), 0.43*Rw*R2/(Cz*den)+0.57/Cz],
                    [R3/(Cm*den)+1/(R1*Cm), Rw*R3/(Cm*den), 0.43*Rw*R3/(Cm*den)]])
    Cct = np.array([[0, 0, 0]])
    Dct = np.array([[0, 0]])
    # convert cont time matrices to discrete time using zoh
    (A, B, C, D, dt) = cont2discrete((Act, Bct, Cct, Dct), ts*60, method='zoh')
    (A, E, C, D, dt) = cont2discrete((Act, Ect, Cct, Dct), ts*60, method='zoh')
    return A, B, E


def powerset(iterable):
    """
        Auxiliary function for computing the power set of an iterable:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        Does not create the set explicitly.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def extract_PGE_loadshape(filename, start_date=None, end_date=None, name=None):
    """
        Reads in tab-separated files (falsely labeled .xls by PG&E) from the
        PG&E website: http://www.pge.com/tariffs/energy_use_prices.shtml
        Parses the dates and replaces all missing values (b/c of DST) by NaN
    """
    loadshapes = pd.read_csv(filename, sep='\t', parse_dates=['Date'],
                             na_values=['.'], index_col='Date')
    if start_date is not None:
        loadshapes = loadshapes[loadshapes.index >= start_date]
    if end_date is not None:
        loadshapes = loadshapes[loadshapes.index <= end_date]
    # get DST transisiton times in US/Pacfic timezone
    ttimes = pd.DatetimeIndex(timezone('US/Pacific')._utc_transition_times[1:],
                              tz=timezone('UTC')).tz_convert('US/Pacific')
    ttimes = ttimes[(ttimes >= loadshapes.index[0]) &
                    (ttimes <= loadshapes.index[-1])]
    # fix inconsistencies b/c of DST changes
    dst_switch_days = loadshapes.index.isin(ttimes.date)
    non_switch_data = loadshapes[np.logical_not(dst_switch_days)].drop(
        'Profile', axis=1)
    switch_data = loadshapes[dst_switch_days].drop('Profile', axis=1)
    idx = pd.DatetimeIndex(
        start=loadshapes.index[0],
        end=loadshapes.index[-1] + pd.Timedelta(hours=23),
        freq='H', tz=timezone('US/Pacific'))
    isswitch = pd.DatetimeIndex(idx.date).isin(ttimes.date)
    nsidx = idx[np.logical_not(isswitch)]
    loadSeries = pd.Series(
        non_switch_data.values.astype(np.float64).flatten(), nsidx)
    dst_series = []
    for day in switch_data.index:
        vals = switch_data.loc[day].values
        tsidx = pd.DatetimeIndex(start=day, end=day+pd.Timedelta(hours=23),
                                 freq='1H', tz=timezone('US/Pacific'))
        if (day.month > 0) & (day.month < 5):
            daydata = np.concatenate([vals[:2], vals[3:]])
        else:
            daydata = np.insert(vals, 1, vals[1])
        dst_series.append(pd.Series(daydata, tsidx))
    loadSeries = pd.concat([loadSeries] + dst_series).sort_index()
    if name is not None:
        loadSeries.name = name
    return loadSeries


def daily_occurrences(index, tz='US/Pacific'):
    """
        Takes in a pandas DateTimeIndex and returns a pandas Series
        indexed by the days in index with the number of occurances of
        timestamps on that day as the values.
    """
    locidx = index.tz_convert(tz)
    occ = pd.DataFrame({'occurences': 1}, index=locidx).groupby(
        locidx.date).count()['occurences']
    occ.index = pd.DatetimeIndex(occ.index)
    return occ


def _parse_nbt_data():
    """
        Prepares data for CAISO net benefits test.
    """
    # define NERC holidays for use in CAISO net benefits test
    NERC_holidays = ([datetime(year, 1, 1) for year in [2012, 2013, 2014]] +
                     [datetime(2012, 5, 28), datetime(2013, 5, 27),
                      datetime(2014, 5, 26)] +
                     [datetime(year, 7, 4) for year in [2012, 2013, 2014]] +
                     [datetime(2012, 9, 3), datetime(2012, 9, 2),
                      datetime(2014, 9, 1)] +
                     [datetime(2014, 11, 22), datetime(2014, 11, 28),
                      datetime(2014, 11, 27)] +
                     [datetime(year, 12, 25) for year in [2012, 2013, 2014]])
    holiday_idx = [
        pd.date_range(
            start=day, end=day+pd.DateOffset(days=1) - pd.Timedelta(minutes=15),
            tz='US/Pacific', freq='15Min')
        for day in NERC_holidays
    ]
    NERC_hd_ts = pd.DatetimeIndex.union_many(holiday_idx[0], holiday_idx[1:])
    NERC_hd_ts = NERC_hd_ts.sort_values().tz_convert('GMT')
    # load the values of the CAISO nbt from data file
    nbl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'CAISO_NBT.csv')
    nbt = pd.read_csv(nbl_file, parse_dates=['Month'], index_col='Month')
    # convert into a single dataframe with all tiestamps present
    dfs = []
    for start, end in zip(nbt.index,
                          nbt.index.shift(1, pd.DateOffset(months=1))):
        tsstart = start.tz_localize('US/Pacific').tz_convert('GMT')
        tsend = (end.tz_localize('US/Pacific').tz_convert('GMT') -
                 pd.Timedelta(minutes=15))
        dfs.append(pd.DataFrame(
            {
                'OnPeak': nbt.loc[start]['OnPeak'],
                'OffPeak': nbt.loc[start]['OffPeak']
            },
            index=pd.date_range(start=tsstart, end=tsend, freq='15Min')))
    nbt = pd.concat(dfs)
    # get indices which count as "OnPeak" as defined by CAISO
    locidx = nbt.index.tz_convert('US/Pacific')
    isNERCholiday = locidx.isin(NERC_hd_ts)
    isPeak = (~isNERCholiday & (locidx.dayofweek < 6) &
              (locidx.hour >= 7) & (locidx.hour < 22))
    # create series with appropriate NBT price level
    nbt = isPeak * nbt['OnPeak'] + ~isPeak * nbt['OffPeak']
    return nbt


def _parse_pdp_days():
    """ Parses data file for PDP peak days """
    pdpd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'PDP_days.csv')
    pdpd = pd.DatetimeIndex(
        pd.read_csv(pdpd_file, parse_dates=[0]).iloc[:, 0])
    return pdpd


def _PGE_tariff_data():
    """
        Prepares data for PGE tariffs.
        nrg   : Energy charges
        dem   : Demand charges
        meter : meter charges?
    """  # values from March 2015 rate change
    nrg = {'Zero':                     {'Summer': {'Peak':        0.0,
                                                   'PartialPeak': 0.0,
                                                   'OffPeak':     0.0},
                                        'Winter': {'PartialPeak': 0.0,
                                                   'OffPeak':     0.0}},
           'A1':                       {'Summer': {'Peak':        0.24176,
                                                   'PartialPeak': 0.24176,
                                                   'OffPeak':     0.24176},
                                        'Winter': {'PartialPeak': 0.16445,
                                                   'OffPeak':     0.16445}},
           'A1TOU':                    {'Summer': {'Peak':        0.26241,
                                                   'PartialPeak': 0.25308,
                                                   'OffPeak':     0.22468},
                                        'Winter': {'PartialPeak': 0.17479,
                                                   'OffPeak':     0.15497}},
           'A1_non-gen':               {'Summer': {'Peak':        0.12813,
                                                   'PartialPeak': 0.12813,
                                                   'OffPeak':     0.12813},
                                        'Winter': {'PartialPeak': 0.09738,
                                                   'OffPeak':     0.09738}},
           'A1TOU_non-gen':            {'Summer': {'Peak':        0.12813,
                                                   'PartialPeak': 0.12813,
                                                   'OffPeak':     0.12813},
                                        'Winter': {'PartialPeak': 0.09738,
                                                   'OffPeak':     0.09738}},
           'A6TOU':                    {'Summer': {'Peak':        0.61173,
                                                   'PartialPeak': 0.28551,
                                                   'OffPeak':     0.15804},
                                        'Winter': {'PartialPeak': 0.18082,
                                                   'OffPeak':     0.14804}},
           'A6TOU_non-gen':            {'Summer': {'Peak':        0.27547,
                                                   'PartialPeak': 0.15083,
                                                   'OffPeak':     0.09938},
                                        'Winter': {'PartialPeak': 0.08612,
                                                   'OffPeak':     0.09088}},
           'A10_secondary':            {'Summer': {'Peak':        0.16116,
                                                   'PartialPeak': 0.16116,
                                                   'OffPeak':     0.16116},
                                        'Winter': {'PartialPeak': 0.11674,
                                                   'OffPeak':     0.11674}},
           'A10_secondary_non-gen':    {'Summer': {'Peak':        0.06012,
                                                   'PartialPeak': 0.06012,
                                                   'OffPeak':     0.06012},
                                        'Winter': {'PartialPeak': 0.04025,
                                                   'OffPeak':     0.04025}},
           'A10_primary':              {'Summer': {'Peak':        0.14936,
                                                   'PartialPeak': 0.14936,
                                                   'OffPeak':     0.14936},
                                        'Winter': {'PartialPeak': 0.11069,
                                                   'OffPeak':     0.11069}},
           'A10_transmission':         {'Summer': {'Peak':        0.12137,
                                                   'PartialPeak': 0.12137,
                                                   'OffPeak':     0.12137},
                                        'Winter': {'PartialPeak': 0.09583,
                                                   'OffPeak':     0.09583}},
           'A10TOU_secondary':         {'Summer': {'Peak':        0.17891,
                                                   'PartialPeak': 0.17087,
                                                   'OffPeak':     0.14642},
                                        'Winter': {'PartialPeak': 0.12750,
                                                   'OffPeak':     0.10654}},
           'A10TOU_secondary_non-gen': {'Summer': {'Peak':        0.06012,
                                                   'PartialPeak': 0.06012,
                                                   'OffPeak':     0.06012},
                                        'Winter': {'PartialPeak': 0.04025,
                                                   'OffPeak':     0.04025}},
           'A10TOU_primary':           {'Summer': {'Peak':        0.20249,
                                                   'PartialPeak': 0.15193,
                                                   'OffPeak':     0.1253},
                                        'Winter': {'PartialPeak': 0.12857,
                                                   'OffPeak':     0.11269}},
           'A10TOU_transmission':      {'Summer': {'Peak':        0.13481,
                                                   'PartialPeak': 0.12958,
                                                   'OffPeak':     0.10973},
                                        'Winter': {'PartialPeak': 0.10392,
                                                   'OffPeak':     0.08816}},
           'E19TOU_secondary':         {'Summer': {'Peak':        0.16233,
                                                   'PartialPeak': 0.10893,
                                                   'OffPeak':     0.07397},
                                        'Winter': {'PartialPeak': 0.10185,
                                                   'OffPeak':     0.07797}},
           'E19TOU_primary':           {'Summer': {'Peak':        0.14861,
                                                   'PartialPeak': 0.10219,
                                                   'OffPeak':     0.07456},
                                        'Winter': {'PartialPeak': 0.09696,
                                                   'OffPeak':     0.07787}},
           'E19TOU_transmission':      {'Summer': {'Peak':        0.09129,
                                                   'PartialPeak': 0.08665,
                                                   'OffPeak':     0.07043},
                                        'Winter': {'PartialPeak': 0.085,
                                                   'OffPeak':     0.07214}},
           'E19TOU_secondary_non-gen': {'Summer': {'Peak':        0.02627,
                                                   'PartialPeak': 0.02627,
                                                   'OffPeak':     0.02627},
                                        'Winter': {'PartialPeak': 0.02627,
                                                   'OffPeak':     0.02627}},
           'OptFlat_non-gen':          {'Summer': {'Peak':        0.0,
                                                   'PartialPeak': 0.0,
                                                   'OffPeak':     0.0},
                                        'Winter': {'PartialPeak': 0.0,
                                                   'OffPeak':     0.0}}}
    dem = {'A10_primary':               {'Summer': 15.22, 'Winter': 8.20},
           'A10_secondary':             {'Summer': 16.23, 'Winter': 8.00},
           'A10_secondary_non-gen':     {'Summer': 11.89, 'Winter': 8.00},
           'A10_transmission':          {'Summer': 10.85, 'Winter': 6.29},
           'A10TOU_primary':            {'Summer': 15.22, 'Winter': 8.20},
           'A10TOU_secondary':          {'Summer': 16.23, 'Winter': 8.00},
           'A10TOU_secondary_non-gen':  {'Summer': 11.89, 'Winter': 8.00},
           'A10TOU_transmission':       {'Summer': 10.85, 'Winter': 6.29},
           'E19TOU_secondary':          {'Summer': {'mpeak': 19.04,
                                                    'ppeak': 4.24,
                                                    'max': 15.07},
                                         'Winter': {'ppeak': 0.24,
                                                    'max': 15.07}},
           'E19TOU_primary':            {'Summer': {'mpeak': 18.91,
                                                    'ppeak': 4.06,
                                                    'max': 12.08},
                                         'Winter': {'ppeak': 0.46,
                                                    'max': 12.08}},
           'E19TOU_transmission':       {'Summer': {'mpeak': 17.03,
                                                    'ppeak': 3.78,
                                                    'max': 7.87},
                                         'Winter': {'ppeak': 0.0,
                                                    'max': 7.87}},
           'E19TOU_secondary_non-gen':  {'Summer': {'mpeak': 5.49,
                                                    'ppeak': 1.48,
                                                    'max': 15.07},
                                         'Winter': {'ppeak': 0.24,
                                                    'max': 15.07}}
           }  # All data here from the March 2015 rate change,
    # to accomodate non-gen rates, based on March 2015 tariffs, from
    # http://www.pge.com/includes/docs/pdfs/mybusiness/customerservice/
    # energychoice/communitychoiceaggregation/faq/
    # business_non_generation_rates.pdf.
    meter = {'A1':    0.65708,
             'A1TOU': 0.65708,
             'A6TOU': 0.65708 + 0.20107,
             'A10_secondary':    4.59959,
             'A10_primary':      4.59959,
             'A10_transmission': 4.59959,
             'A10TOU_secondary':    4.59959,
             'A10TOU_primary':      4.59959,
             'A10TOU_transmission': 4.59959,
             'E19TOU_secondary': {'Mandatory': {'S': 19.71253,
                                                'P': 32.85421,
                                                'T': 59.13758},
                                  'Voluntary': {'SmartMeter': 4.59959,
                                                'NoSmartMeter': 4.777}},
             'E19TOU_primary': {'Mandatory': {'S': 19.71253,
                                              'P': 32.85421,
                                              'T': 59.13758},
                                'Voluntary': {'SmartMeter': 4.59959,
                                              'NoSmartMeter': 4.777}},
             'E19TOU_transmission': {'Mandatory': {'S': 19.71253,
                                                   'P': 32.85421,
                                                   'T': 59.13758},
                                     'Voluntary': {'SmartMeter': 4.59959,
                                                   'NoSmartMeter': 4.777}},
             'Zero': 0.0
             }
    return nrg, dem, meter


def _PGE_tariff_data_2015():
    """
        Prepares data for PGE tariffs.
        nrg   : Energy charges
        dem   : Demand charges
        meter : meter charges?
    """
    nrg = {'Zero':                {'Summer': {'Peak':        0.0,
                                              'PartialPeak': 0.0,
                                              'OffPeak':     0.0},
                                   'Winter': {'PartialPeak': 0.0,
                                              'OffPeak':     0.0}},
           'A1':                  {'Summer': {'Peak':        0.23977,
                                              'PartialPeak': 0.23977,
                                              'OffPeak':     0.23977},
                                   'Winter': {'PartialPeak': 0.16246,
                                              'OffPeak':     0.16246}},
           'A1TOU':               {'Summer': {'Peak':        0.26042,
                                              'PartialPeak': 0.25109,
                                              'OffPeak':     0.22269},
                                   'Winter': {'PartialPeak': 0.17280,
                                              'OffPeak':     0.15298}},
           'A6TOU':               {'Summer': {'Peak':        0.60974,
                                              'PartialPeak': 0.28352,
                                              'OffPeak':     0.15605},
                                   'Winter': {'PartialPeak': 0.17883,
                                              'OffPeak':     0.14605}},
           'A10_secondary':       {'Summer': {'Peak':        0.16116,
                                              'PartialPeak': 0.16116,
                                              'OffPeak':     0.16116},
                                   'Winter': {'PartialPeak': 0.11674,
                                              'OffPeak':     0.11674}},
           'A10_primary':         {'Summer': {'Peak':        0.14936,
                                              'PartialPeak': 0.14936,
                                              'OffPeak':     0.14936},
                                   'Winter': {'PartialPeak': 0.11069,
                                              'OffPeak':     0.11069}},
           'A10_transmission':    {'Summer': {'Peak':        0.12137,
                                              'PartialPeak': 0.12137,
                                              'OffPeak':     0.12137},
                                   'Winter': {'PartialPeak': 0.09583,
                                              'OffPeak':     0.09583}},
           'A10TOU_secondary':    {'Summer': {'Peak':        0.17891,
                                              'PartialPeak': 0.17087,
                                              'OffPeak':     0.14642},
                                   'Winter': {'PartialPeak': 0.12750,
                                              'OffPeak':     0.10654}},
           'A10TOU_primary':      {'Summer': {'Peak':        0.16420,
                                              'PartialPeak': 0.15846,
                                              'OffPeak':     0.13650},
                                   'Winter': {'PartialPeak': 0.11949,
                                              'OffPeak':     0.10231}},
           'A10TOU_transmission': {'Summer': {'Peak':        0.13481,
                                              'PartialPeak': 0.12958,
                                              'OffPeak':     0.10973},
                                   'Winter': {'PartialPeak': 0.10392,
                                              'OffPeak':     0.08816}},
           'E19TOU_secondary':    {'Summer': {'Peak':        0.16233,
                                              'PartialPeak': 0.10893,
                                              'OffPeak':     0.07397},
                                   'Winter': {'PartialPeak': 0.10185,
                                              'OffPeak':     0.07797}},
           'E19TOU_primary':      {'Summer': {'Peak':        0.14861,
                                              'PartialPeak': 0.10219,
                                              'OffPeak':     0.07456},
                                   'Winter': {'PartialPeak': 0.09696,
                                              'OffPeak':     0.07787}},
           'E19TOU_transmission': {'Summer': {'Peak':        0.09129,
                                              'PartialPeak': 0.08665,
                                              'OffPeak':     0.07043},
                                   'Winter': {'PartialPeak': 0.08500,
                                              'OffPeak':     0.07214}}}
    dem = {'A10_primary':         {'Summer': 15.54, 'Winter': 7.31},
           'A10_secondary':       {'Summer': 14.53, 'Winter': 7.51},
           'A10_transmission':    {'Summer': 10.16, 'Winter': 5.60},
           'A10TOU_primary':      {'Summer': 15.54, 'Winter': 7.31},
           'A10TOU_secondary':    {'Summer': 14.53, 'Winter': 7.51},
           'A10TOU_transmission': {'Summer': 10.16, 'Winter': 5.60},
           'E19TOU_secondary':    {'Summer': {'mpeak': 19.04,
                                              'ppeak': 4.42,
                                              'max': 14.38},
                                   'Winter': {'ppeak': 0.24,
                                              'max': 14.38}},
           'E19TOU_primary':      {'Summer': {'mpeak': 18.91,
                                              'ppeak': 4.06,
                                              'max': 11.39},
                                   'Winter': {'ppeak': 0.46,
                                              'max': 11.39}},
           'E19TOU_transmission': {'Summer': {'mpeak': 17.03,
                                              'ppeak': 3.78,
                                              'max': 7.18},
                                   'Winter': {'ppeak': 0.0,
                                              'max': 7.18}}
           }
    meter = {'A1':    0.65708,
             'A1TOU': 0.65708,
             'A6TOU': 0.65708 + 0.20107,
             'A10_secondary':    4.59959,
             'A10_primary':      4.59959,
             'A10_transmission': 4.59959,
             'A10TOU_secondary':    4.59959,
             'A10TOU_primary':      4.59959,
             'A10TOU_transmission': 4.59959,
             'E19TOU_secondary': {'Mandatory': {'S': 19.71253,
                                                'P': 32.85421,
                                                'T': 59.13758},
                                  'Voluntary': {'SmartMeter': 4.59959,
                                                'NoSmartMeter': 4.777}},
             'E19TOU_primary': {'Mandatory': {'S': 19.71253,
                                              'P': 32.85421,
                                              'T': 59.13758},
                                'Voluntary': {'SmartMeter': 4.59959,
                                              'NoSmartMeter': 4.777}},
             'E19TOU_transmission': {'Mandatory': {'S': 19.71253,
                                                   'P': 32.85421,
                                                   'T': 59.13758},
                                     'Voluntary': {'SmartMeter': 4.59959,
                                                   'NoSmartMeter': 4.777}},
             'Zero': 0.0
             }
    return nrg, dem, meter


def _PGE_tariff_data_2014():
    """
        Prepares data for PGE tariffs.
        nrg   : Energy charges
        dem   : Demand charges
        meter : meter charges?
    """
    nrg = {'Zero':                  {'Summer': {'Peak':      0.0,
                                              'PartialPeak': 0.0,
                                              'OffPeak':     0.0},
                                   'Winter': {'PartialPeak': 0.0,
                                              'OffPeak':     0.0}},
           'A1':                  {'Summer': {'Peak':        0.21706,
                                              'PartialPeak': 0.21706,
                                              'OffPeak':     0.21706},
                                   'Winter': {'PartialPeak': 0.15014,
                                              'OffPeak':     0.15014}},
           'A1TOU':               {'Summer': {'Peak':        0.23592,
                                              'PartialPeak': 0.22764,
                                              'OffPeak':     0.20244},
                                   'Winter': {'PartialPeak': 0.15944,
                                              'OffPeak':     0.14138}},
           'A1_non-gen':          {'Summer': {'Peak':        0.12803,
                                              'PartialPeak': 0.12803,
                                              'OffPeak':     0.12803},
                                   'Winter': {'PartialPeak': 0.09728,
                                              'OffPeak':     0.09728}},
           'A1TOU_non-gen':       {'Summer': {'Peak':        0.12803,
                                              'PartialPeak': 0.12803,
                                              'OffPeak':     0.12803},
                                   'Winter': {'PartialPeak': 0.09728,
                                                 'OffPeak':  0.09728}},
           'A6TOU':               {'Summer': {'Peak':        0.54053,
                                              'PartialPeak': 0.25139,
                                              'OffPeak':     0.14103},
                                   'Winter': {'PartialPeak': 0.16045,
                                              'OffPeak':     0.13103}},
           'A6TOU_non-gen':       {'Summer': {'Peak':        0.27537,
                                              'PartialPeak': 0.15073,
                                              'OffPeak':     0.09928},
                                   'Winter': {'PartialPeak': 0.08602,
                                               'OffPeak':    0.09078}},
           'A10_secondary':       {'Summer': {'Peak':        0.14911,
                                              'PartialPeak': 0.14911,
                                              'OffPeak':     0.14911},
                                   'Winter': {'PartialPeak': 0.11077,
                                              'OffPeak':     0.11077}},
           'A10_secondary_non-gen':{'Summer': {'Peak':       0.06002,
                                              'PartialPeak': 0.06002,
                                              'OffPeak':     0.06002},
                                   'Winter': {'PartialPeak': 0.04015,
                                              'OffPeak':     0.04015}},
           'A10_primary':         {'Summer': {'Peak':        0.13939,
                                              'PartialPeak': 0.13939,
                                              'OffPeak':     0.13939},
                                   'Winter': {'PartialPeak': 0.10552,
                                              'OffPeak':     0.10552}},
           'A10_transmission':    {'Summer': {'Peak':        0.11550,
                                              'PartialPeak': 0.11550,
                                              'OffPeak':     0.11550},
                                   'Winter': {'PartialPeak': 0.09255,
                                              'OffPeak':     0.09255}},
           'A10TOU_secondary':    {'Summer': {'Peak':        0.16506,
                                              'PartialPeak': 0.15784,
                                              'OffPeak':     0.13588},
                                   'Winter': {'PartialPeak': 0.12044,
                                              'OffPeak':     0.10161}},
           'A10TOU_secondary_non-gen':{'Summer': {'Peak':    0.06002,
                                              'PartialPeak': 0.06002,
                                              'OffPeak':     0.06002},
                                   'Winter': {'PartialPeak': 0.04015,
                                              'OffPeak':     0.04015}},
           'A10TOU_primary':      {'Summer': {'Peak':        0.15271,
                                              'PartialPeak': 0.14756,
                                              'OffPeak':     0.12785},
                                   'Winter': {'PartialPeak': 0.11342,
                                              'OffPeak':     0.09800}},
           'A10TOU_transmission': {'Summer': {'Peak':        0.12759,
                                              'PartialPeak': 0.12288,
                                              'OffPeak':     0.10505},
                                   'Winter': {'PartialPeak': 0.09981,
                                              'OffPeak':     0.08566}},
           'E19TOU_secondary':    {'Summer': {'Peak':        0.15255,
                                              'PartialPeak': 0.10461,
                                              'OffPeak':     0.07321},
                                   'Winter': {'PartialPeak': 0.09824,
                                              'OffPeak':     0.07681}},
           'E19TOU_primary':      {'Summer': {'Peak':        0.14013,
                                              'PartialPeak': 0.09849,
                                              'OffPeak':     0.07369},
                                   'Winter': {'PartialPeak': 0.0938,
                                              'OffPeak':     0.07666}},
           'E19TOU_transmission': {'Summer': {'Peak':        0.0887,
                                              'PartialPeak': 0.08453,
                                              'OffPeak':     0.06998},
                                   'Winter': {'PartialPeak': 0.08305,
                                              'OffPeak':     0.07152}},
           'E19TOU_secondary_non-gen': {'Summer': {'Peak':        0.02618,
                                                   'PartialPeak': 0.02618,
                                                   'OffPeak':     0.02618},
                                        'Winter': {'PartialPeak': 0.02618,
                                                   'OffPeak':     0.02618}},
           'OptFlat_non-gen':          {'Summer': {'Peak':        0.0,
                                                   'PartialPeak': 0.0,
                                                   'OffPeak':     0.0},
                                        'Winter': {'PartialPeak': 0.0,
                                                   'OffPeak':     0.0}}}
    dem = {'A10_primary':         {'Summer': 12.61, 'Winter': 6.48},
           'A10_secondary':       {'Summer': 13.36, 'Winter': 6.26},
           'A10_secondary_non-gen':{'Summer': 11.89, 'Winter': 8.00},
           'A10_transmission':    {'Summer': 8.94, 'Winter': 4.84},
           'A10TOU_primary':      {'Summer': 12.61, 'Winter': 6.48},
           'A10TOU_secondary':    {'Summer': 13.36, 'Winter': 6.26},
           'A10TOU_secondary_non-gen': {'Summer': 11.89, 'Winter': 8.00},
           'A10TOU_transmission': {'Summer': 8.94, 'Winter': 4.84},
           'E19TOU_secondary':    {'Summer': {'mpeak': 16.78,
                                              'ppeak': 3.87,
                                              'max': 12.24},
                                   'Winter': {'ppeak': 0.21,
                                              'max': 12.24}},
           'E19TOU_secondary_non-gen': {'Summer': {'mpeak': 5.49,
                                           'ppeak': 1.48,
                                           'max': 15.07},
                                'Winter': {'ppeak': 0.24,
                                           'max': 15.07}},
           'E19TOU_primary':      {'Summer': {'mpeak': 16.67,
                                              'ppeak': 3.56,
                                              'max': 9.72},
                                   'Winter': {'ppeak': 0.38,
                                              'max': 9.72}},
           'E19TOU_transmission': {'Summer': {'mpeak': 15.28,
                                              'ppeak': 3.38,
                                              'max': 5.95},
                                   'Winter': {'ppeak': 0.0,
                                              'max': 5.95}}
           }
    meter = {'A1':    0.32854,
             'A1TOU': 0.32854,
             'A6TOU': 0.32854 + 0.20107,
             'A10_secondary':    4.59959,
             'A10_primary':      4.59959,
             'A10_transmission': 4.59959,
             'A10TOU_secondary':    4.59959,
             'A10TOU_primary':      4.59959,
             'A10TOU_transmission': 4.59959,
             'E19TOU_secondary': {'Mandatory': {'S': 19.71253,
                                                'P': 32.85421,
                                                'T': 59.13758},
                                  'Voluntary': {'SmartMeter': 4.59959,
                                                'NoSmartMeter': 4.777}},
             'E19TOU_primary': {'Mandatory': {'S': 19.71253,
                                              'P': 32.85421,
                                              'T': 59.13758},
                                'Voluntary': {'SmartMeter': 4.59959,
                                              'NoSmartMeter': 4.777}},
             'E19TOU_transmission': {'Mandatory': {'S': 19.71253,
                                                   'P': 32.85421,
                                                   'T': 59.13758},
                                     'Voluntary': {'SmartMeter': 4.59959,
                                                   'NoSmartMeter': 4.777}}
             }
    return nrg, dem, meter



def _PGE_tariff_data_2013():
    """
        Prepares data for PGE tariffs.
        nrg   : Energy charges
        dem   : Demand charges
        meter : meter charges?
    """
    nrg = {'Zero':                  {'Summer': {'Peak':      0.0,
                                              'PartialPeak': 0.0,
                                              'OffPeak':     0.0},
                                   'Winter': {'PartialPeak': 0.0,
                                              'OffPeak':     0.0}},
           'A1':                  {'Summer': {'Peak':        0.21015,
                                              'PartialPeak': 0.21015,
                                              'OffPeak':     0.21015},
                                   'Winter': {'PartialPeak': 0.14671,
                                              'OffPeak':     0.14671}},
           'A1TOU':               {'Summer': {'Peak':        0.22769,
                                              'PartialPeak': 0.22007,
                                              'OffPeak':     0.19690},
                                   'Winter': {'PartialPeak': 0.15541,
                                              'OffPeak':     0.13865}},
           'A1_non-gen':          {'Summer': {'Peak':        0.12813,
                                              'PartialPeak': 0.12813,
                                              'OffPeak':     0.12813},
                                   'Winter': {'PartialPeak': 0.09738,
                                              'OffPeak':     0.09738}},
           'A1TOU_non-gen':       {'Summer': {'Peak':        0.12813,
                                              'PartialPeak': 0.12813,
                                              'OffPeak':     0.12813},
                                   'Winter': {'PartialPeak': 0.09738,
                                                 'OffPeak':  0.09738}},
           'A6TOU':               {'Summer': {'Peak':        0.48657,
                                              'PartialPeak': 0.23713,
                                              'OffPeak':     0.13768},
                                   'Winter': {'PartialPeak': 0.15534,
                                              'OffPeak':     0.12768}},
           'A6TOU_non-gen':       {'Summer': {'Peak':        0.27547,
                                              'PartialPeak': 0.15083,
                                              'OffPeak':     0.09938},
                                   'Winter': {'PartialPeak': 0.08612,
                                               'OffPeak':    0.09088}},
           'A10_secondary':       {'Summer': {'Peak':        0.14335,
                                              'PartialPeak': 0.14335,
                                              'OffPeak':     0.14335},
                                   'Winter': {'PartialPeak': 0.10578,
                                              'OffPeak':     0.10578}},
           'A10_secondary_non-gen': {'Summer': {'Peak':      0.06012,
                                                'PartialPeak':0.06012,
                                                'OffPeak':   0.06012},
                                     'Winter': {'PartialPeak':0.04025,
                                                'OffPeak':   0.04025}},
           'A10_primary':         {'Summer': {'Peak':        0.13373,
                                              'PartialPeak': 0.13373,
                                              'OffPeak':     0.13373},
                                   'Winter': {'PartialPeak': 0.10085,
                                              'OffPeak':     0.10085}},
           'A10_transmission':    {'Summer': {'Peak':        0.10959,
                                              'PartialPeak': 0.10959,
                                              'OffPeak':     0.10959},
                                   'Winter': {'PartialPeak': 0.08816,
                                              'OffPeak':     0.08816}},
           'A10TOU_secondary':    {'Summer': {'Peak':        0.15821,
                                              'PartialPeak': 0.15148,
                                              'OffPeak':     0.13102},
                                   'Winter': {'PartialPeak': 0.11479,
                                              'OffPeak':     0.09724}},
           'A10TOU_secondary_non-gen': {'Summer': {'Peak':   0.06012,
                                                'PartialPeak':0.06012,
                                                'OffPeak':   0.06012},
                                     'Winter': {'PartialPeak':0.04025,
                                                'OffPeak':   0.04025}},
           'A10TOU_primary':      {'Summer': {'Peak':        0.14613,
                                              'PartialPeak': 0.14133,
                                              'OffPeak':     0.12299},
                                   'Winter': {'PartialPeak': 0.10820,
                                              'OffPeak':     0.09385}},
           'A10TOU_transmission': {'Summer': {'Peak':        0.12087,
                                              'PartialPeak': 0.11648,
                                              'OffPeak':     0.09983},
                                   'Winter': {'PartialPeak': 0.09494,
                                              'OffPeak':     0.08173}},
           'E19TOU_secondary':    {'Summer': {'Peak':        0.14364,
                                              'PartialPeak': 0.09896,
                                              'OffPeak':     0.0697},
                                   'Winter': {'PartialPeak': 0.09303,
                                              'OffPeak':     0.07305}},
           'E19TOU_primary':      {'Summer': {'Peak':        0.13195,
                                              'PartialPeak': 0.09318,
                                              'OffPeak':     0.07009},
                                   'Winter': {'PartialPeak': 0.08881,
                                              'OffPeak':     0.07286}},
           'E19TOU_transmission': {'Summer': {'Peak':        0.08392,
                                              'PartialPeak': 0.08005,
                                              'OffPeak':     0.06654},
                                   'Winter': {'PartialPeak': 0.07868,
                                              'OffPeak':     0.06797}},
           'E19TOU_secondary_non-gen': {'Summer': {'Peak':        0.02627,
                                                   'PartialPeak': 0.02627,
                                                   'OffPeak':     0.02627},
                                        'Winter': {'PartialPeak': 0.02627,
                                                   'OffPeak':     0.02627}},
           'OptFlat_non-gen':          {'Summer': {'Peak':        0.0,
                                                   'PartialPeak': 0.0,
                                                   'OffPeak':     0.0},
                                        'Winter': {'PartialPeak': 0.0,
                                                   'OffPeak':     0.0}}}
    dem = {'A10_primary':         {'Summer': 11.78, 'Winter': 5.81},
           'A10_secondary':       {'Summer': 12.57, 'Winter': 5.60},
           'A10_secondary_non-gen': {'Summer': 11.89, 'Winter': 8.00},
           'A10TOU_secondary_non-gen': {'Summer': 11.89, 'Winter': 8.00},
           'A10_transmission':    {'Summer': 7.94, 'Winter': 4.11},
           'A10TOU_primary':      {'Summer': 11.78, 'Winter': 5.81},
           'A10TOU_secondary':    {'Summer': 12.57, 'Winter': 5.60},
           'A10TOU_transmission': {'Summer': 7.94, 'Winter': 4.11},
           'E19TOU_secondary':    {'Summer': {'mpeak': 16.13,
                                              'ppeak': 3.74,
                                              'max': 11.79},
                                   'Winter': {'ppeak': 0.21,
                                              'max': 11.79}},
           'E19TOU_secondary_non-gen': {'Summer': {'mpeak': 5.49,
                                           'ppeak': 1.48,
                                           'max': 15.07},
                                'Winter': {'ppeak': 0.24,
                                           'max': 15.07}},
           'E19TOU_primary':      {'Summer': {'mpeak': 15.96,
                                              'ppeak': 3.43,
                                              'max': 9.18},
                                   'Winter': {'ppeak': 0.40,
                                              'max': 9.18}},
           'E19TOU_transmission': {'Summer': {'mpeak': 14.19,
                                              'ppeak': 3.14,
                                              'max': 5.31},
                                   'Winter': {'ppeak': 0.0,
                                              'max': 5.31}}
           }
    meter = {'A1':    0.32854,
             'A1TOU': 0.32854,
             'A6TOU': 0.32854 + 0.20107,
             'A10_secondary':    4.59959,
             'A10_primary':      4.59959,
             'A10_transmission': 4.59959,
             'A10TOU_secondary':    4.59959,
             'A10TOU_primary':      4.59959,
             'A10TOU_transmission': 4.59959,
             'E19TOU_secondary': {'Mandatory': {'S': 19.71253,
                                                'P': 32.85421,
                                                'T': 59.13758},
                                  'Voluntary': {'SmartMeter': 4.59959,
                                                'NoSmartMeter': 4.777}},
             'E19TOU_primary': {'Mandatory': {'S': 19.71253,
                                              'P': 32.85421,
                                              'T': 59.13758},
                                'Voluntary': {'SmartMeter': 4.59959,
                                              'NoSmartMeter': 4.777}},
             'E19TOU_transmission': {'Mandatory': {'S': 19.71253,
                                                   'P': 32.85421,
                                                   'T': 59.13758},
                                     'Voluntary': {'SmartMeter': 4.59959,
                                                   'NoSmartMeter': 4.777}}
             }
    return nrg, dem, meter



def _PGE_tariff_data_2012():
    """
        Prepares data for PGE tariffs.
        nrg   : Energy charges
        dem   : Demand charges
        meter : meter charges?
    """
    nrg = {'Zero':                  {'Summer': {'Peak':      0.0,
                                              'PartialPeak': 0.0,
                                              'OffPeak':     0.0},
                                   'Winter': {'PartialPeak': 0.0,
                                              'OffPeak':     0.0}},
           'A1':                  {'Summer': {'Peak':        0.20459,
                                              'PartialPeak': 0.20459,
                                              'OffPeak':     0.20459},
                                   'Winter': {'PartialPeak': 0.14430,
                                              'OffPeak':     0.14430}},
           'A1TOU':               {'Summer': {'Peak':        0.21915,
                                              'PartialPeak': 0.21258,
                                              'OffPeak':     0.19259},
                                   'Winter': {'PartialPeak': 0.1516,
                                              'OffPeak':     0.13753}},
           'A1_non-gen':          {'Summer': {'Peak':        0.12850,
                                              'PartialPeak': 0.12850,
                                              'OffPeak':     0.12850},
                                   'Winter': {'PartialPeak': 0.09775,
                                              'OffPeak':     0.09775}},
           'A1TOU_non-gen':       {'Summer': {'Peak':        0.12850,
                                              'PartialPeak': 0.12850,
                                              'OffPeak':     0.12850},
                                   'Winter': {'PartialPeak': 0.09775,
                                                 'OffPeak':  0.09775}},
           'A6TOU':               {'Summer': {'Peak':        0.43932,
                                              'PartialPeak': 0.22435,
                                              'OffPeak':     0.13777},
                                   'Winter': {'PartialPeak': 0.15184,
                                              'OffPeak':     0.12777}},
           'A6TOU_non-gen':       {'Summer': {'Peak':        0.27584,
                                              'PartialPeak': 0.15120,
                                              'OffPeak':     0.09975},
                                   'Winter': {'PartialPeak': 0.08649,
                                               'OffPeak':    0.09125}},
           'A10_secondary':       {'Summer': {'Peak':        0.13771,
                                              'PartialPeak': 0.13771,
                                              'OffPeak':     0.13771},
                                   'Winter': {'PartialPeak': 0.10268,
                                              'OffPeak':     0.10268}},
           'A10_secondary_non-gen': {'Summer': {'Peak':      0.06050,
                                                'PartialPeak':0.06050,
                                                'OffPeak':   0.06050},
                                     'Winter': {'PartialPeak':0.04063,
                                                'OffPeak':   0.04063}},
           'A10_primary':         {'Summer': {'Peak':        0.12881,
                                              'PartialPeak': 0.12881,
                                              'OffPeak':     0.12881},
                                   'Winter': {'PartialPeak': 0.09841,
                                              'OffPeak':     0.09841}},
           'A10_transmission':    {'Summer': {'Peak':        0.10474,
                                              'PartialPeak': 0.10474,
                                              'OffPeak':     0.10474},
                                   'Winter': {'PartialPeak': 0.08606,
                                              'OffPeak':     0.08606}},
           'A10TOU_secondary':    {'Summer': {'Peak':        0.15067,
                                              'PartialPeak': 0.14480,
                                              'OffPeak':     0.12696},
                                   'Winter': {'PartialPeak': 0.11053,
                                              'OffPeak':     0.09523}},
           'A10TOU_secondary_non-gen': {'Summer': {'Peak':   0.06050,
                                                'PartialPeak':0.06050,
                                                'OffPeak':   0.06050},
                                     'Winter': {'PartialPeak':0.04063,
                                                'OffPeak':   0.04063}},
           'A10TOU_primary':      {'Summer': {'Peak':        0.13963,
                                              'PartialPeak': 0.13544,
                                              'OffPeak':     0.11945},
                                   'Winter': {'PartialPeak': 0.10482,
                                              'OffPeak':     0.09230}},
           'A10TOU_transmission': {'Summer': {'Peak':        0.11458,
                                              'PartialPeak': 0.11076,
                                              'OffPeak':     0.09623},
                                   'Winter': {'PartialPeak': 0.09197,
                                              'OffPeak':     0.08045}},
           'E19TOU_secondary':    {'Summer': {'Peak':        0.13413,
                                              'PartialPeak': 0.09516,
                                              'OffPeak':     0.06965},
                                   'Winter': {'PartialPeak': 0.09,
                                              'OffPeak':     0.07257}},
           'E19TOU_primary':      {'Summer': {'Peak':        0.1237,
                                              'PartialPeak': 0.0899,
                                              'OffPeak':     0.06976},
                                   'Winter': {'PartialPeak': 0.08608,
                                              'OffPeak':     0.07217}},
           'E19TOU_transmission': {'Summer': {'Peak':        0.08178,
                                              'PartialPeak': 0.0784,
                                              'OffPeak':     0.06662},
                                   'Winter': {'PartialPeak': 0.07721,
                                              'OffPeak':     0.06787}},
           'E19TOU_secondary_non-gen': {'Summer': {'Peak':        0.02659,
                                                   'PartialPeak': 0.02659,
                                                   'OffPeak':     0.02659},
                                        'Winter': {'PartialPeak': 0.02659,
                                                   'OffPeak':     0.02659}},
           'OptFlat_non-gen':          {'Summer': {'Peak':        0.0,
                                                   'PartialPeak': 0.0,
                                                   'OffPeak':     0.0},
                                        'Winter': {'PartialPeak': 0.0,
                                                   'OffPeak':     0.0}}}
    dem = {'A10_primary':         {'Summer': 11.38, 'Winter': 5.84},
           'A10_secondary':       {'Summer': 12.15, 'Winter': 5.63},
           'A10_secondary_non-gen': {'Summer': 11.89, 'Winter': 8.00},
           'A10TOU_secondary_non-gen': {'Summer': 11.89, 'Winter': 8.00},
           'A10_transmission':    {'Summer': 7.47, 'Winter': 4.13},
           'A10TOU_primary':      {'Summer': 11.38, 'Winter': 5.84},
           'A10TOU_secondary':   {'Summer': 12.15, 'Winter': 5.63},
           'A10TOU_transmission': {'Summer': 7.47, 'Winter': 4.13},
           'E19TOU_secondary':    {'Summer': {'mpeak': 14.70,
                                              'ppeak': 3.43,
                                              'max': 11.85},
                                   'Winter': {'ppeak': 0.21,
                                              'max': 11.85}},
           'E19TOU_primary':      {'Summer': {'mpeak': 14.48,
                                              'ppeak': 3.15,
                                              'max': 9.23},
                                   'Winter': {'ppeak': 0.40,
                                              'max': 9.23}},
           'E19TOU_transmission': {'Summer': {'mpeak': 12.37,
                                              'ppeak': 2.74,
                                              'max': 5.35},
                                   'Winter': {'ppeak': 0.0,
                                              'max': 5.35}},
           'E19TOU_secondary_non-gen': {'Summer': {'mpeak': 5.49,
                                           'ppeak': 1.48,
                                           'max': 15.07},
                                'Winter': {'ppeak': 0.24,
                                           'max': 15.07}},
           }
    meter = {'A1':    0.32854,
             'A1TOU': 0.32854,
             'A6TOU': 0.32854 + 0.20107,
             'A10_secondary':    4.59959,
             'A10_primary':      4.59959,
             'A10_transmission': 4.59959,
             'A10TOU_secondary':    4.59959,
             'A10TOU_primary':      4.59959,
             'A10TOU_transmission': 4.59959,
             'E19TOU_secondary': {'Mandatory': {'S': 19.71253,
                                                'P': 32.85421,
                                                'T': 59.13758},
                                  'Voluntary': {'SmartMeter': 4.59959,
                                                'NoSmartMeter': 4.777}},
             'E19TOU_primary': {'Mandatory': {'S': 19.71253,
                                              'P': 32.85421,
                                              'T': 59.13758},
                                'Voluntary': {'SmartMeter': 4.59959,
                                              'NoSmartMeter': 4.777}},
             'E19TOU_transmission': {'Mandatory': {'S': 19.71253,
                                                   'P': 32.85421,
                                                   'T': 59.13758},
                                     'Voluntary': {'SmartMeter': 4.59959,
                                                   'NoSmartMeter': 4.777}}
             }
    return nrg, dem, meter



def _pdp_credits():
    ''' peak demand credits data
        would like to integrate this above, but didnt want to mess with
        code if you call the above somehwere else
        pdpkwh : dictionary with pdp per [kwh] energy credits; keys are tarrifs
        pdpdem : dictionary with pdp demand credits; keys are tarrifs
        pdpchg : dictionary with pdp charges; keys are tarrifs
    '''
    pdpkwh = {'A1TOU':               {'Summer': {'Peak':        0.01016,
                                                 'PartialPeak': 0.01016,
                                                 'OffPeak':     0.01016},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'A6TOU':               {'Summer': {'Peak':        0.1217,
                                                 'PartialPeak': 0.02434,
                                                 'OffPeak':     0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'A10TOU_secondary':    {'Summer': {'Peak':        0.00641,
                                                 'PartialPeak': 0.00641,
                                                 'OffPeak':     0.00641},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'A10TOU_primary':      {'Summer': {'Peak':        0.00608,
                                                 'PartialPeak': 0.00608,
                                                 'OffPeak':     0.00608},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'A10TOU_transmission': {'Summer': {'Peak':        0.00344,
                                                 'PartialPeak': 0.00344,
                                                 'OffPeak':     0.00344},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'E19TOU_secondary':    {'Summer': {'Peak':        0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak':     0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'E19TOU_primary':      {'Summer': {'Peak':        0.00,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak':     0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}},
              'E19TOU_transmission': {'Summer': {'Peak':        0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak':     0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak':     0.0}}
              }
    pdpdem = {'A1TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A6TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A10TOU_secondary':    {'Summer': 2.89, 'Winter': 0.0},
              'A10TOU_primary':      {'Summer': 2.74, 'Winter': 0.0},
              'A10TOU_transmission': {'Summer': 3.04, 'Winter': 0.0},
              'E19TOU_secondary':    {'Summer': {'peak': 6.19, 'ppeak': 1.34},
                                      'Winter': 0.0},
              'E19TOU_primary':      {'Summer': {'peak': 5.99, 'ppeak': 1.16},
                                      'Winter': 0.0},
              'E19TOU_transmission': {'Summer': {'peak': 5.51, 'ppeak': 1.22},
                                      'Winter': 0.0}
              }
    pdpchg = {'A1TOU':  0.60,
              'A6TOU':  1.20,
              'A10TOU': 0.90,
              'E19TOU': 1.20
              }
    return pdpkwh, pdpdem, pdpchg



def _pdp_credits_2014():
    ''' peak demand credits data
        would like to integrate this above, but didnt want to mess with
        code if you call the above somehwere else
        pdpkwh : dictionary with pdp per [kwh] energy credits; keys are tarrifs
        pdpdem : dictionary with pdp demand credits; keys are tarrifs
        pdpchg : dictionary with pdp charges; keys are tarrifs
    '''
    pdpkwh = {'A1TOU':               {'Summer': {'Peak': 0.00977,
                                                 'PartialPeak': 0.00977,
                                                 'OffPeak': 0.00977},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A6TOU':               {'Summer': {'Peak': 0.1083,
                                                 'PartialPeak': 0.02166,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_secondary':    {'Summer': {'Peak': 0.00702,
                                                 'PartialPeak': 0.00702,
                                                 'OffPeak': 0.00702},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_primary':      {'Summer': {'Peak': 0.008,
                                                 'PartialPeak': 0.008,
                                                 'OffPeak': 0.008},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_transmission': {'Summer': {'Peak': 0.00861,
                                                 'PartialPeak': 0.00861,
                                                 'OffPeak': 0.00861},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_secondary':    {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_primary':      {'Summer': {'Peak': 0.00,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_transmission': {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}}
              }
    pdpdem = {'A1TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A6TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A10TOU_secondary':    {'Summer': 2.60, 'Winter': 0.0},
              'A10TOU_primary':      {'Summer': 2.46, 'Winter': 0.0},
              'A10TOU_transmission': {'Summer': 2.73, 'Winter': 0.0},
              'E19TOU_secondary':    {'Summer': {'peak': 6.37, 'ppeak': 1.38},
                                      'Winter': 0.0},
              'E19TOU_primary':      {'Summer': {'peak': 6.04, 'ppeak': 1.17},
                                      'Winter': 0.0},
              'E19TOU_transmission': {'Summer': {'peak': 6.29, 'ppeak': 1.39},
                                      'Winter': 0.0}
              }
    pdpchg = {
        'A1TOU':  0.60,
        'A6TOU':  1.20,
        'A10TOU_secondary': 0.90,
        'A10TOU_primary': 0.90,
        'A10TOU_transmission': 0.90,
        'E19TOU_secondary': 1.20,
        'E19TOU_primary': 1.20,
        'E19TOU_transmission': 1.20,
    }
    return pdpkwh, pdpdem, pdpchg

def _pdp_credits_2013():
    ''' peak demand credits data
        would like to integrate this above, but didnt want to mess with
        code if you call the above somehwere else
        pdpkwh : dictionary with pdp per [kwh] energy credits; keys are tarrifs
        pdpdem : dictionary with pdp demand credits; keys are tarrifs
        pdpchg : dictionary with pdp charges; keys are tarrifs
    '''
    pdpkwh = {'A1TOU':               {'Summer': {'Peak': 0.00944,
                                                 'PartialPeak': 0.00944,
                                                 'OffPeak': 0.00944},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A6TOU':               {'Summer': {'Peak': 0.09522,
                                                 'PartialPeak': 0.01904,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_secondary':    {'Summer': {'Peak': 0.00721,
                                                 'PartialPeak': 0.00721,
                                                 'OffPeak': 0.00721},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_primary':      {'Summer': {'Peak': 0.0078,
                                                 'PartialPeak': 0.0078,
                                                 'OffPeak': 0.0078},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_transmission': {'Summer': {'Peak': 0.00838,
                                                 'PartialPeak': 0.00838,
                                                 'OffPeak': 0.00838},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_secondary':    {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_primary':      {'Summer': {'Peak': 0.00,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_transmission': {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}}
              }
    pdpdem = {'A1TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A6TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A10TOU_secondary':    {'Summer': 2.42, 'Winter': 0.0},
              'A10TOU_primary':      {'Summer': 2.29, 'Winter': 0.0},
              'A10TOU_transmission': {'Summer': 2.55, 'Winter': 0.0},
              'E19TOU_secondary':    {'Summer': {'peak': 6.27, 'ppeak': 1.35},
                                      'Winter': 0.0},
              'E19TOU_primary':      {'Summer': {'peak': 5.94, 'ppeak': 1.15},
                                      'Winter': 0.0},
              'E19TOU_transmission': {'Summer': {'peak': 5.63, 'ppeak': 1.25},
                                      'Winter': 0.0}
              }
    pdpchg = {
        'A1TOU':  0.60,
        'A6TOU':  1.20,
        'A10TOU_secondary': 0.90,
        'A10TOU_primary': 0.90,
        'A10TOU_transmission': 0.90,
        'E19TOU_secondary': 1.20,
        'E19TOU_primary': 1.20,
        'E19TOU_transmission': 1.20,
    }
    return pdpkwh, pdpdem, pdpchg


def _pdp_credits_2012():
    ''' peak demand credits data
        would like to integrate this above, but didnt want to mess with
        code if you call the above somehwere else
        pdpkwh : dictionary with pdp per [kwh] energy credits; keys are tarrifs
        pdpdem : dictionary with pdp demand credits; keys are tarrifs
        pdpchg : dictionary with pdp charges; keys are tarrifs
    '''
    pdpkwh = {'A1TOU':               {'Summer': {'Peak': 0.00991,
                                                 'PartialPeak': 0.00991,
                                                 'OffPeak': 0.00991},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A6TOU':               {'Summer': {'Peak': 0.09283,
                                                 'PartialPeak': 0.01857,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_secondary':    {'Summer': {'Peak': 0.00875,
                                                 'PartialPeak': 0.00875,
                                                 'OffPeak': 0.00875},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_primary':      {'Summer': {'Peak': 0.00899,
                                                 'PartialPeak': 0.00899,
                                                 'OffPeak': 0.00899},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'A10TOU_transmission': {'Summer': {'Peak': 0.00648,
                                                 'PartialPeak': 0.00648,
                                                 'OffPeak': 0.00648},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_secondary':    {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_primary':      {'Summer': {'Peak': 0.00,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}},
              'E19TOU_transmission': {'Summer': {'Peak': 0.0,
                                                 'PartialPeak': 0.0,
                                                 'OffPeak': 0.0},
                                      'Winter': {'PartialPeak': 0.0,
                                                 'OffPeak': 0.0}}
              }
    pdpdem = {'A1TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A6TOU':               {'Summer': 0.0,  'Winter': 0.0},
              'A10TOU_secondary':    {'Summer': 2.11, 'Winter': 0.0},
              'A10TOU_primary':      {'Summer': 1.99, 'Winter': 0.0},
              'A10TOU_transmission': {'Summer': 2.23, 'Winter': 0.0},
              'E19TOU_secondary':    {'Summer': {'peak': 6.35, 'ppeak': 1.37},
                                      'Winter': 0.0},
              'E19TOU_primary':      {'Summer': {'peak': 6.09, 'ppeak': 1.18},
                                      'Winter': 0.0},
              'E19TOU_transmission': {'Summer': {'peak': 5.54, 'ppeak': 1.23},
                                      'Winter': 0.0}
              }
    pdpchg = {
        'A1TOU':  0.60,
        'A6TOU':  1.20,
        'A10TOU_secondary': 0.90,
        'A10TOU_primary': 0.90,
        'A10TOU_transmission': 0.90,
        'E19TOU_secondary': 1.20,
        'E19TOU_primary': 1.20,
        'E19TOU_transmission': 1.20,
    }
    return pdpkwh, pdpdem, pdpchg




def get_pdp_demand_credit(tariff, month, year):
    ''' Only for E19 '''
    if year is not None:
        pdpdem_credit = pdpdem_credit_yearly[year]
    if (month >= 5) & (month <= 10):
        return pdpdem_credit[tariff]['Summer']
    else:
        return pdpdem_credit[tariff]['Winter']


def get_demand_charge(tariff, month, isPDP=False, year=None):
    """ Return demand charge. Note that the output of this function
        is a scalar for all tariffs except for E19, but a dict for E19.
    """
    # if (month >= 5) & (month <= 10):
    #    return dem_charges[tariff]['Summer']
    # else:
    #    return dem_charges[tariff]['Winter']
    if year is not None:
        dem_charges = dem_charges_yearly[year]
        pdpdem_credit = pdpdem_credit_yearly[year]
    if not(isPDP):
        if not('E19' in tariff):
            if (month >= 5) & (month <= 10):
                return dem_charges[tariff]['Summer']
            else:
                return dem_charges[tariff]['Winter']
        else:
            ''' This is incase we need to adjust what is output for E19
                right now E19 outputs a dictionary
            '''
            if (month >= 5) & (month <= 10):
                return dem_charges[tariff]['Summer']
            else:
                return dem_charges[tariff]['Winter']
    else:
        # if pdp remove demand credits
        if not('E19' in tariff):
            if (month >= 5) & (month <= 10):
                return (dem_charges[tariff]['Summer'] -
                        pdpdem_credit[tariff]['Summer'])
            else:
                return (dem_charges[tariff]['Winter'] -
                        pdpdem_credit[tariff]['Winter'])

        else:
            # This is incase we need to adjust what is output for E19
            # E19 Pdp demand charges to be added in blopt for now -- Lily
            if (month >= 5) & (month <= 10):
                return dem_charges[tariff]['Summer']
            else:
                return dem_charges[tariff]['Winter']


def get_energy_charges(index, tariff, isRT=False, LMP=None,
                       isPDP=False, carbon=False, year=None):
    """
        Return energy charges for each element in a pandas DateTimeIndex as a
        pandas DataFrame, indexed by a Datetimeindex localized to GMT timezone.
        Requires that the passed index is also timezone-aware.
    """
    # do some checks whether the tariff makes sense
    year = index[0].year
    if year is not None:
        nrg_charges = nrg_charges_yearly[year]
        pdpkwh_credit = pdpkwh_credit_yearly[year]
        pdp_charges = pdpchg_chrg_yearly[year]

    if isRT:
        if isPDP:
            raise Exception('Cannot combine RTP with PDP.')
        if 'Zero' in tariff:
            tar = nrg_charges['Zero']
        else:
            if tariff not in non_gen_tariffs:
                raise Exception('Tariff {} '.format(tariff) +
                                'is not compatible with RTP.')
            else:
                tar = nrg_charges[tariff + '_non-gen']
    else:
        if 'OptFlat' in tariff:  # want to do Opt Flat LMPmG
            if 'non-gen' in tariff:
                optflat = 0.0
            else:
                optflat = LMP.mean() / 1000.0
            tar = {'Summer': {'Peak':        optflat,
                              'PartialPeak': optflat,
                              'OffPeak':     optflat},
                   'Winter': {'PartialPeak': optflat,
                              'OffPeak':     optflat}}
        else:
            tar = nrg_charges[tariff]
    if isPDP:
        if tariff not in pdp_compatible:
            raise Exception('Tariff {} not '.format(tariff) +
                            'compatible with PDP.')
        else:
            pdpcr = pdpkwh_credit[tariff]
            pdpchrg = pdp_charges[tariff]
    idx = index.tz_convert('US/Pacific')
    iswknd = idx.dayofweek > 5
    holidays = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
    iswknd = iswknd | pd.DatetimeIndex(idx.date).isin(holidays)
    issummer = (idx.month >= 5) & (idx.month <= 10)
    ToD = idx.hour + idx.minute / 60
    ispeak = ~iswknd & issummer & (ToD >= 12) & (ToD < 18)
    ispartial_summer = (~iswknd & issummer & (((ToD >= 8.5) & (ToD < 12)) |
                        ((ToD >= 18) & (ToD < 21.5))))
    ispartial_winter = ~iswknd & ~issummer & ((ToD >= 8.5) & (ToD < 21.5))
    isoff_summer = issummer & ~(ispeak | ispartial_summer)
    isoff_winter = ~issummer & ~ispartial_winter
    dfs = []
    for time, i in zip(['Peak', 'PartialPeak', 'OffPeak'],
                       [idx[ispeak], idx[ispartial_summer],
                        idx[isoff_summer]]):
        if isPDP:
            dfs.append(pd.DataFrame(
                {'EnergyCharge': [tar['Summer'][time] -
                                  pdpcr['Summer'][time]]*len(i)},
                index=i))
        else:
            dfs.append(pd.DataFrame(
                {'EnergyCharge': [tar['Summer'][time]]*len(i)},
                index=i))
    for time, i in zip(['PartialPeak', 'OffPeak'],
                       [idx[ispartial_winter], idx[isoff_winter]]):
        dfs.append(pd.DataFrame({'EnergyCharge': [tar['Winter'][time]]*len(i)},
                                index=i))
    chronRates = pd.concat(dfs, axis=0).sort_index()
    if isPDP:
        cidx = chronRates.index
        pdpind = ((cidx.hour >= 12) & (cidx.hour < 18) &
                  (cidx.normalize().isin(pdp_days.tz_localize('US/Pacific'))))
        chronRates.loc[pdpind, 'EnergyCharge'] += pdpchrg
    chronRates = chronRates.tz_convert('GMT')
    if isRT:
        chronRates['EnergyCharge'] += LMP.loc[index[0]:index[-1]] / 1000.0
    if carbon:
        chronRates['EnergyCharge'] += pd.Series(
            carbon_costs).loc[idx.year].values / 1000.0
    return chronRates


def get_DR_rewards(LMP, isLMPmG=False, tariff=None):
    """
        Helper function to determine the reward for DR reductions. If
        isLMPmG is True, reward reductions by LMP - G, i.e. by the LMP
        minus the generation component of the respective tariff:
        LMP - G = LMP - (tariff_w_gen - tariff_wout_gen)
                = LMP + tariff_wout_gen - tariff_w_gen
                = RTP_tariff - tariff_w_gen
    """
    # convert LMPs to LMP-Gs if option is specified in kwargs:
    if (isLMPmG is None) or not isLMPmG:
        return LMP
    elif tariff is None:
        raise Exception('Must provide a tariff when using LMP-G rewards.')
    elif tariff not in non_gen_tariffs:
        raise Exception('Tariff {} is not '.format(tariff) +
                        'compatible with LMP-G DR compensation.')
    elif 'OptFlat' in tariff:
        return LMP - LMP.mean()
    else:
        rtp_chrgs = get_energy_charges(
            LMP.index, tariff, isRT=True, LMP=LMP)['EnergyCharge']
        wgen_chrgs = get_energy_charges(
            LMP.index, tariff, isRT=False)['EnergyCharge']
        return 1000 * (rtp_chrgs - wgen_chrgs)


def net_benefits_test(LMP, n='all', how='absolute', maxperday=24,
                      ignore_days=0):
    """
        Takes a time-series of LMPs ($/MW) indexed by a timezone-aware
        pd.Datetimeindex and returns a boolean vector indicating
        whether the LMP exceeds the CAISO Net Benefits Test threshold.
        Also allows to specify option n, which results in returning
        only those n intervals with the highest relative price levels.
        Finally, if maxperday < 24, select at most maxperday events
        in a single day.
    """
    # ignore some days in the beginning (to make BL computation feasible)
    if ignore_days > 0:
        LMP_nbt = LMP[LMP.index >= LMP.index[0] + pd.Timedelta(
            days=ignore_days)]
    else:
        LMP_nbt = LMP
    if how == 'absolute':
        criterion = LMP_nbt - nbt.loc[LMP_nbt.index]
    elif how == 'relative':
        criterion = LMP_nbt / nbt.loc[LMP_nbt.index]
    if not maxperday < 24:
        if n == 'all':
            return LMP > nbt.loc[LMP_nbt.index]
        else:
            idcs = criterion.nlargest(n).index
            return pd.Series(LMP.index.isin(idcs), index=LMP.index)
    else:
        idcs = pd.DatetimeIndex([], tz='US/Pacific')
        if n == 'all':
            for d, v in criterion.tz_convert('US/Pacific').groupby(
                    pd.TimeGrouper('D')):
                idcs.append(v.nlargest(maxperday).index)
        else:
            countdict = {}
            ncurr = 0
            for ts, val in criterion.tz_convert('US/Pacific').sort_values(
                    ascending=False).iteritems():
                if ts.date() in countdict:
                    if countdict[ts.date()] == maxperday:
                        continue
                    else:
                        idcs = idcs.append(pd.DatetimeIndex([ts]))
                        countdict[ts.date()] += 1
                        ncurr += 1
                else:
                    idcs = idcs.append(pd.DatetimeIndex([ts]))
                    countdict[ts.date()] = 1
                    ncurr += 1
                if ncurr == n:
                    break
        return pd.Series(LMP.index.isin(idcs.tz_convert('GMT')),
                         index=LMP.index)


# parse data for CAISO net benefits test
nbt = _parse_nbt_data()
# define PGE tariff data
nrg_charges, dem_charges, meter_charges = _PGE_tariff_data()
nrg_charges_yearly, dem_charges_yearly, meter_charges_yearly = {}, {}, {}
nrg_charges_yearly[2012], dem_charges_yearly[2012], meter_charges_yearly[2012] = _PGE_tariff_data_2012()
nrg_charges_yearly[2013], dem_charges_yearly[2013], meter_charges_yearly[2013] = _PGE_tariff_data_2013()
nrg_charges_yearly[2014], dem_charges_yearly[2014], meter_charges_yearly[2014] = _PGE_tariff_data_2014()

pdpkwh_credit, pdpdem_credit, pdpchg_chrg = _pdp_credits()
pdpkwh_credit_yearly, pdpdem_credit_yearly, pdpchg_chrg_yearly = {}, {}, {}
pdpkwh_credit_yearly[2012], pdpdem_credit_yearly[2012], pdpchg_chrg_yearly[2012] = _pdp_credits_2012()
pdpkwh_credit_yearly[2013], pdpdem_credit_yearly[2013], pdpchg_chrg_yearly[2013] = _pdp_credits_2013()
pdpkwh_credit_yearly[2014], pdpdem_credit_yearly[2014], pdpchg_chrg_yearly[2014] = _pdp_credits_2014()

pdp_days = _parse_pdp_days()
pdp_charges = {'A1TOU': 0.60, 'A6TOU': 1.20, 'A10TOU_secondary': 0.90,
               'A10TOU_primary': 0.90, 'A10TOU_transmission': 0.90,
               'E19TOU_secondary': 1.20, 'E19TOU_primary': 1.20,
               'E19TOU_transmission': 1.20}
# define tariffs that are compatible with RTP
non_gen_tariffs = ['A1', 'A1TOU', 'A6TOU', 'A10_secondary', 'A10TOU_secondary',
                   'E19TOU_secondary', 'OptFlatA1', 'OptFlatA6TOU', 'OptFlat']
pdp_compatible = ['A1TOU', 'A6TOU',
                  'A10TOU_secondary', 'A10TOU_primary', 'A10TOU_transmission',
                  'E19TOU_secondary', 'E19TOU_primary', 'E19TOU_transmission']
