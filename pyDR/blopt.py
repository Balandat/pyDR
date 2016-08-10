"""
Code for the optimization and gaming component of the Baselining work.

@author: Maximilian Balandat, Lillian Ratliff
@date Mar 2, 2016
"""

import numpy as np
import pandas as pd
import logging
from gurobipy import GRB, Model, quicksum, LinExpr
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime

from .utils import (get_energy_charges, get_demand_charge, dem_charges,
                    get_pdp_demand_credit, get_DR_rewards, powerset, E19,
                    carbon_costs)

# define some string formatters
psform = '%Y-%m-%d %H:%M'
dsform = '%Y-%m-%d'


class BLModel(object):
    """
        Abstract base class for Baselining models.
    """

    def __init__(self, name):
        """
            Construct an abstract dynamical system object based on the
            gurobipy Model object 'model'.
        """
        self._name = name
        self._model = Model()

    def get_model(self):
        """
            Returns the underlying gurobiy Model object.
        """
        return self._model

    def set_dynsys(self, dynsys):
        """
            Initialize dynamical system for underlying dynamics.
        """
        self._dynsys = dynsys

    def set_window(self, index):
        """
            Set the window for the optimization. Here index is a pandas
            DatetimeIndex.
        """
        self._index = index
        self._dynsys.set_window(index)

    def energy_charges(self, tariff, isRT=False, LMP=None, isPDP=False,
                       twindow=None, carbon=False):
        """
            Return total enery consumption charges (as determined by the
            tariff's energy charge) as a gurobipy LinExpr.
        """
        locidx = self._index.tz_convert('US/Pacific')
        if isRT and isPDP:
            raise Exception('Cannot combine RTP and PDP.')
        nrg_charges = get_energy_charges(
            self._index, tariff, isRT=isRT, LMP=LMP,
            isPDP=isPDP, carbon=carbon)['EnergyCharge']
        cons = self._dynsys.get_consumption()['energy']
        if twindow is None:
            # echrg_= quicksum([ec * con for ec, con in
            #            zip(nrg_charges.values, cons.values)])
            echrg_ = [ec * con for ec, con in
                      zip(nrg_charges.values, cons.values)]
            echrg = pd.Series(echrg_, index=locidx)
        else:
            nrg_charges_ = nrg_charges.loc[twindow[0]:twindow[1]]
            cons_ = cons.loc[twindow[0]:twindow[1]]
            # echrg = quicksum([ec * con for ec, con in
            #             zip(nrg_charges_.values, cons_.values)])
            echrg_ = [ec * con for ec, con in
                      zip(nrg_charges_.values, cons_.values)]
            indx = locidx[locidx.get_loc(twindow[0]):
                          locidx.get_loc(twindow[1])+1]
            echrg = pd.Series(echrg_, index=indx)
        return echrg

    def demand_charges(self, tariff, isPDP=False):
        """
            Return the total demand charges under the tariff as a
            gurobipy LinExpr.
        """
        # determine which year/month combinations there is a demand charge,
        # and create a variable for each of them
        if hasattr(self, '_maxcon'):
            for maxcon in self._maxcon.values():
                self._model.remove(maxcon)
            del self._maxcon
        if hasattr(self, '_maxconbnd'):
            for maxconbnd in self._maxconbnd.values():
                self._model.remove(maxconbnd)
            del self._maxconbnd
        if hasattr(self, '_maxconppk'):
            for maxconppk in self._maxconppk.values():
                self._model.remove(maxconppk)
            del self._maxconppk
        if hasattr(self, '_maxconppkbnd'):
            for maxconppkbnd in self._maxconppkbnd.values():
                self._model.remove(maxconppkbnd)
            del self._maxconppkbnd
        if hasattr(self, '_maxconpk'):
            for maxconpk in self._maxconpk.values():
                self._model.remove(maxconpk)
            del self._maxconpk
        if hasattr(self, '_maxconpkbnd'):
            for maxconpkbnd in self._maxconpkbnd.values():
                self._model.remove(maxconpkbnd)
            del self._maxconpkbnd
        if hasattr(self, '_maxconpks'):
            for maxconpks in self._maxconpks.values():
                self._model.remove(maxconpks)
            del self._maxconpks
        if hasattr(self, '_maxconppkw'):
            for maxconppkw in self._maxconppkw.values():
                self._model.remove(maxconppkw)
            del self._maxconppkw
        if hasattr(self, '_maxconppkbndw'):
            for maxconppkbndw in self._maxconppkbndw.values():
                self._model.remove(maxconppkbndw)
            del self._maxconppkbndw
        if hasattr(self, '_maxconppks'):
            for maxconppks in self._maxconppks.values():
                self._model.remove(maxconppks)
            del self._maxconppks
        if hasattr(self, '_maxconppkbnds'):
            for maxconppkbnds in self._maxconppkbnds.values():
                self._model.remove(maxconppkbnds)
            del self._maxconppkbnds
        self._model.update()
        locidx = self._index.tz_convert('US/Pacific')
        ym_dict = {year: np.unique(locidx[locidx.year == year].month)
                   for year in np.unique(locidx.year)}
        indx = []
        for year, months in ym_dict.items():
            for month in months:
                indx.append(pd.Timestamp(datetime(year, month, 1),
                                         tz='US/Pacific'))
        if tariff in dem_charges:
            if not(tariff in E19):
                self._maxcon, self._maxconbnd = {}, {}
                # locidx = self._index.tz_convert('US/Pacific')
                # print locidx
                # the following creates a dictionary with all years in the data
                # as keys, and for each year the value is an array of (unique)
                # months that appear during that year. This is used for keeping
                # track of the peak consumpiton for the demand charge
                # ym_dict = {year: np.unique(locidx[locidx.year == year].month)
                #           for year in np.unique(locidx.year)}
                # indx=[]
                for year, months in ym_dict.items():
                    for month in months:
                        # declare variable for max consumption
                        self._maxcon[year, month] = self._model.addVar(
                            vtype=GRB.CONTINUOUS,
                            name='maxcon[{},{}]'.format(year, month))
                        # indx.append(pd.Timestamp(datetime(year,month,1),tz='US/Pacific'))
                self._model.update()
                # now add in the necessary constraints and update objective
                dcharges = []

                cons = self._dynsys.get_consumption()['power']
                for year, months in ym_dict.items():
                    for month in months:
                        relcons = cons[(locidx.year == year) &
                                       (locidx.month == month)].values
                        for i, con in enumerate(relcons):
                            self._maxconbnd[year, month, i] = self._model.addConstr(
                                lhs=self._maxcon[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=con, name='maxconbnd[{},{},{}]'.format(
                                    year, month, i))
                        # dcharges += (get_demand_charge(tariff, month, isPDP)*
                        #              self._maxcon[year, month])
                        dcharges.append(
                            (get_demand_charge(tariff, month, isPDP) *
                             self._maxcon[year, month]))
                dcharges = pd.Series(dcharges, index=indx)
                self._model.update()
                return dcharges
            else:
                # for E19 tarrifs
                idx_ = self._index.tz_convert('US/Pacific')
                iswknd = idx_.dayofweek > 5
                holidays = USFederalHolidayCalendar().holidays(
                    idx_.min(), idx_.max())
                iswknd = iswknd | pd.DatetimeIndex(idx_.date).isin(holidays)
                issummer = (idx_.month >= 5) & (idx_.month <= 10)
                ToD = idx_.hour + idx_.minute / 60
                ispeak = ~iswknd & issummer & (ToD >= 12) & (ToD < 18)
                ispartial_summer = (~iswknd & issummer & (
                    ((ToD >= 8.5) & (ToD < 12)) |
                    ((ToD >= 18) & (ToD < 21.5))))
                ispartial_winter = ~iswknd & ~issummer & (
                    (ToD >= 8.5) & (ToD < 21.5))
                # create dictionaries for variables
                self._maxcon, self._maxconbnd = {}, {}
                self._maxconppks, self._maxconppkbnds = {}, {}
                self._maxconpks, self._maxconpkbnds = {}, {}
                self._maxconpk, self._maxconpkbnd = {}, {}
                self._maxconppk, self._maxconppkbnd = {}, {}
                self._maxconppkw, self._maxconppkbndw = {}, {}
                # locidx = self._index.tz_convert('US/Pacific')
                # ym_dict = {year: np.unique(locidx[locidx.year == year].month)
                #            for year in np.unique(locidx.year)}
                # indx=[]
                for year, months in ym_dict.items():
                    for month in months:
                        # declare variable for max consumption
                        self._maxcon[year, month] = self._model.addVar(
                            vtype=GRB.CONTINUOUS,
                            name='maxcon[{},{}]'.format(year, month))
                        # declare variable for part peak consumption
                        self._maxconppk[year, month] = self._model.addVar(
                            vtype=GRB.CONTINUOUS,
                            name='maxconppk[{},{}]'.format(year, month))
                        # declare variable for max peak only in summer
                        if (5 <= month) & (month <= 10):
                            # add variable for maximum peak usage in summer
                            self._maxconpk[year, month] = self._model.addVar(
                                vtype=GRB.CONTINUOUS,
                                name='maxconpk[{},{}]'.format(year, month))
                        # indx.append(pd.Timestamp(datetime(year,month,1),tz='US/Pacific'))
                self._model.update()  # update model
                # now add in the necessary constraints and update objective
                dcharges = []
                cons = self._dynsys.get_consumption()['power']
                for year, months in ym_dict.items():
                    for month in months:
                        dchrg = 0.0
                        # for peak summer less than max demand
                        if (month >= 5) & (month <= 10):
                            self._maxconpkbnd[year, month] = self._model.addConstr(
                                lhs=self._maxcon[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=self._maxconpk[year, month],
                                name='maxconpkbnd[{},{}]'.format(year, month))
                        # max partial peak summer greater than consumption
                        ppconsum = cons[(ispartial_summer) &
                                        (locidx.year == year) &
                                        (locidx.month == month)].values
                        for i, con in enumerate(ppconsum):
                            self._maxconppkbnds[year, month, i] = self._model.addConstr(
                                lhs=self._maxconppk[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=con,
                                name='maxconppkbnds[{},{},{}]'.format(
                                    year, month, i))
                        # max peak consumption summer
                        pconsum = cons[(ispeak) & (locidx.year == year) &
                                       (locidx.month == month)].values
                        for i, con in enumerate(pconsum):
                            self._maxconpkbnds[year, month, i] = self._model.addConstr(
                                lhs=self._maxconpk[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=con,
                                name='maxconpkbnds[{},{},{}]'.format(
                                    year, month, i))
                        # max partial peak winter
                        ppkconwin = cons[(ispartial_winter) &
                                         (locidx.year == year) &
                                         (locidx.month == month)].values
                        for i, con in enumerate(ppkconwin):
                            self._maxconppkbndw[year, month, i] = self._model.addConstr(
                                lhs=self._maxconppk[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=con,
                                name='maxconppkbndw[{},{},{}]'.format(
                                    year, month, i))
                        # max demand each month
                        relcons = cons[(locidx.year == year) &
                                       (locidx.month == month)].values
                        for i, con in enumerate(relcons):
                            self._maxconbnd[year, month, i] = self._model.addConstr(
                                lhs=self._maxcon[year, month],
                                sense=GRB.GREATER_EQUAL,
                                rhs=con, name='maxconbnd[{},{},{}]'.format(
                                    year, month, i))
                        # max partial peaks (summer & winter) < than max demand
                        self._maxconppkbnd[year, month, i] = self._model.addConstr(
                            lhs=self._maxcon[year, month],
                            sense=GRB.GREATER_EQUAL,
                            rhs=self._maxconppk[year, month],
                            name='maxconppkbnd[{},{},{}]'.format(
                                year, month, i))
                        demchrg = get_demand_charge(tariff, month)
                        if (month >= 5) & (month <= 10):
                            mpeakchg = demchrg['mpeak']
                            ppeakchg = demchrg['ppeak']
                            maxchg = demchrg['max']
                            if isPDP:
                                pdpcred = get_pdp_demand_credit(tariff, month)
                                mpeakchg = mpeakchg - pdpcred['peak']
                            dchrg += mpeakchg * self._maxconpk[year, month]
                            # dcharges.append(mpeakchg * self._maxconpk[year, month])
                        else:
                            ppeakchg = demchrg['ppeak']
                            maxchg = demchrg['max']
                        # add partpeak and maximum demand charge
                        dcharges.append(
                            (maxchg * self._maxcon[year, month] +
                             ppeakchg * self._maxconppk[year, month])+dchrg)
                self._model.update()
                dcharges = pd.Series(dcharges, index=indx)
                return dcharges
        else:
            return pd.Series([LinExpr(0.0) for ij in
                              range(0, np.size(indx, 0))], index=indx)

    def DR_compensation(self, LMP, dr_periods, BL='CAISO', **kwargs):
        """
            Return compensation for DR, i.e. reductions w.r.t. baseline.
            Here LMP is a pandas Series (indexed by a tz-aware pandas
            Datetimeindex containing all of the object's indices) and
            dr_hours is a pandas DatetimeIndex.
        """
        # start by removing all variables (might be inefficient, but o/w it
        # is a pain in the ass do deal with the multihour baselines etc.)
        self._removeOld()
        # no work if no DR events are specified
        if (LMP is None) or (dr_periods is None):
            return pd.Series([0.0], index=['None'])
        # get DR rewards (in case we want LMP-G instead of LMP)
        DR_rewards = get_DR_rewards(LMP, isLMPmG=kwargs.get('isLMPmG'),
                                    tariff=kwargs.get('tariff'))
        # populate optimization problem for proper BL choices
        if BL == 'CAISO':
            # print self._DR_comp_CAISO(DR_rewards, dr_periods)
            return self._DR_comp_CAISO(DR_rewards, dr_periods)
        elif BL == 'expMA':
            return self._DR_comp_expMA(DR_rewards, dr_periods, **kwargs)
        else:
            raise NotImplementedError(
                'Baseline type "{}" not known!'.format(BL))

    def _DR_comp_CAISO(self, LMP, dr_periods):
        """
            Return compensation for DR, i.e. reductions w.r.t. CAISO baseline.
            Here LMP is a pandas Series (indexed by a tz-aware pandas
            Datetimeindex containing all of the object's indices) and
            dr_hours is a pandas DatetimeIndex. Note that LMP may also be
            LMP-G, i.e. the LMP minus the generation component of the tariff.
        """
        valid_periods = dr_periods[dr_periods.isin(self._index)].tz_convert(
            'US/Pacific')
        locidx = self._index.tz_convert('US/Pacific')
        grouped = valid_periods.groupby(valid_periods.date)
        # define auxiliary variables for each possible dr period if none exist
        self._red, self._z, self._bl = {}, {}, {}
        self._redpos, self._redBL, self._red0, self._blcon = {}, {}, {}, {}
        self._dr_periods = valid_periods

        # add variables if there are days w/ multiple possible DR events
        if np.max([len(grp) for grp in grouped.values()]) > 1:
            self._zday, self._zdaysum, self._zdaymax = {}, {}, {}
        # now create variables for different days and periods within each day
        for day, periods in grouped.iteritems():
            daystr = day.strftime(dsform)
            perstrs = [per.strftime(psform) for per in periods]
            if len(periods) > 1:
                self._zday[daystr] = self._model.addVar(
                    vtype=GRB.BINARY, name='zday[{}]'.format(daystr))
            for period, perstr in zip(periods, perstrs):
                self._red[perstr] = self._model.addVar(
                    vtype=GRB.CONTINUOUS, name='red[{}]'.format(perstr))
                self._z[perstr] = self._model.addVar(
                    vtype=GRB.BINARY, name='z[{}]'.format(perstr))
                self._bl[perstr] = self._model.addVar(
                    vtype=GRB.CONTINUOUS, name='bl[{}]'.format(perstr))
        self._model.update()  # this must be done before defining constaints
        # determine "bigM" value from the bounds on the control variables
        M = np.sum(np.asarray(self._dynsys._opts['nrg_coeffs']) *
                   (self._dynsys._opts['umax'] - self._dynsys._opts['umin']),
                   axis=1).max()
        # if u is not bounded the the above results in an NaN value. We need
        # to deal with this in a better way than the following:
        if np.isnan(M):
            M = 1e9
        # perform some preparations for the constraints
        # drcomp = 0.0
        nrgcons = self._dynsys.get_consumption()['energy']
        lmps = LMP.tz_convert('US/Pacific').loc[locidx] / 1000  # to $/kWh
        holidays = USFederalHolidayCalendar().holidays(
            start=locidx.min(), end=locidx.max())
        isBusiness = (locidx.dayofweek < 5) & (~locidx.isin(holidays))
        isBusiness = pd.Series(isBusiness, index=locidx)
        # add constraints on varible zday (if multiple periods per day)
        for day, periods in grouped.iteritems():
            daystr = day.strftime(dsform)
            perstrs = [per.strftime(psform) for per in periods]
            if len(periods) > 1:
                self._zdaysum[daystr] = self._model.addConstr(
                    lhs=self._zday[daystr],
                    sense=GRB.LESS_EQUAL,
                    rhs=quicksum([self._z[ps] for ps in perstrs]),
                    name='zdaysum[{}]'.format(daystr))
                for period, perstr in zip(periods, perstrs):
                    self._zdaymax[perstr] = self._model.addConstr(
                        lhs=self._zday[daystr],
                        sense=GRB.GREATER_EQUAL,
                        rhs=self._z[perstr],
                        name='zdaymax[{}]'.format(perstr))
        self._model.update()
        # formulate constaints and add terms to objective
        drcomp_ = []
        for i, day in enumerate(grouped):
            periods = grouped[day]
            # print('Formulating constraints for day {} of {}'.format(
            #     i, len(grouped)))
            perstrs = [per.strftime(psform) for per in periods]
            for period, perstr in zip(periods, perstrs):
                per_select = ((locidx < period) &
                              (locidx.hour == period.hour) &
                              (locidx.minute == period.minute))
                if isBusiness.loc[period]:
                    nmax = 10
                    per_select = per_select & isBusiness.values
                else:
                    nmax = 4
                    per_select = per_select & (~isBusiness.values)
                similars = locidx[per_select].sort_values(ascending=False)
                # now go through similar days sucessively
                sim_nonDR, sim_DR, sim_DR_mult = [], [], []
                for sim in similars:
                    if len(sim_nonDR) == nmax:
                        continue
                    if sim in self._dr_periods:
                        sim_DR += [sim]
                        if len(grouped[sim.date()]) > 1:
                            sim_DR_mult += [sim]
                    else:
                        sim_nonDR += [sim]
                sim_DR = pd.DatetimeIndex(
                    sim_DR).sort_values(ascending=False)
                sim_DR_mult = pd.DatetimeIndex(
                    sim_DR_mult).sort_values(ascending=False)
                sim_nonDR = pd.DatetimeIndex(
                    sim_nonDR).sort_values(ascending=False)
                # get consumption variables
                cons_nonDR = nrgcons.loc[sim_nonDR].values
                # Now add constraits on the baseline variables
                for idxset in powerset(range(len(sim_DR))):
                    K = [sim_DR[i] for i in idxset]
                    Kc = [sim_DR[i] for i in range(len(sim_DR))
                          if i not in idxset]
                    qK = nrgcons.loc[K].values.tolist()
                    # Need to make sure to use zday if there are multiple
                    # events possible that day!
                    zK, zKc = [], []
                    for k in K:
                        if k in sim_DR_mult:
                            zK.append(self._zday[k.strftime(dsform)])
                        else:
                            zK.append(self._z[k.strftime(psform)])
                    for kc in Kc:
                        if kc in sim_DR_mult:
                            zKc.append(self._zday[kc.strftime(dsform)])
                        else:
                            zKc.append(self._z[kc.strftime(psform)])
                    # the following uses that the "closest" days appear first
                    qD = cons_nonDR[:nmax-len(idxset)].tolist()
                    n = len(sim_nonDR)
                    if n == 0:
                        print('No non-DR day available for BL computation -' +
                              ' too many DR events!')
                    bnd = (quicksum(qD + qK) / float(n) +
                           M * quicksum(zK) +
                           M * quicksum([(1-z) for z in zKc]))
                    self._blcon[perstr, idxset] = self._model.addConstr(
                        lhs=self._bl[perstr], sense=GRB.LESS_EQUAL,
                        rhs=bnd, name="blcon[{},{}]".format(perstr, idxset))
                # add constraints on baseline reduction
                self._redpos[perstr] = self._model.addConstr(
                    lhs=self._red[perstr], sense=GRB.GREATER_EQUAL,
                    rhs=0.0, name='redpos[{}]'.format(perstr))
                self._redBL[perstr] = self._model.addConstr(
                    lhs=self._red[perstr], sense=GRB.LESS_EQUAL,
                    rhs=self._bl[perstr] - nrgcons.loc[period],
                    name='redBL[{}]'.format(perstr))
                self._red0[perstr] = self._model.addConstr(
                    lhs=self._red[perstr], sense=GRB.LESS_EQUAL,
                    rhs=self._z[perstr] * M, name='red0[{}]'.format(perstr))
                # add DR compensation to objective
                # drcomp += lmps.loc[period] * self._red[perstr]
                drcomp_.append(lmps.loc[period] * self._red[perstr])
        drcomp = pd.Series(drcomp_, index=self._dr_periods)
        self._model.update()
        return drcomp

    def _DR_comp_expMA(self, LMP, dr_periods, **kwargs):
        """
            Return compensation for DR, i.e. reductions w.r.t. CAISO baseline.
            Here LMP is a pandas Series (indexed by a tz-aware pandas
            Datetimeindex containing all of the object's indices) and
            dr_hours is a pandas DatetimeIndex. Note that LMP may also be
            LMP-G, i.e. the LMP minus the generation component of the tariff.
        """
        # set default values for alphas if not passed as kwargs
        if 'alpha_b' in kwargs:
            alpha_b = kwargs['alpha_b']
        else:
            alpha_b = 0.175  # business day
        if 'alpha_nb' in kwargs:
            alpha_nb = kwargs['alpha_nb']
        else:
            alpha_nb = 0.25  # non-business day
        valid_periods = dr_periods[dr_periods.isin(self._index)]
        locidx = self._index.tz_convert('US/Pacific')
        grouped = valid_periods.groupby(
            valid_periods.tz_convert('US/Pacific').date)
        # define auxiliary variables for each possible dr period if none exist
        self._red, self._z, self._bl = {}, {}, {}
        self._redpos, self._redBL, self._red0, self._blcon = {}, {}, {}, {}
        self._dr_periods = valid_periods
        # add variables if there are days w/ multiple possible DR events
        if np.max([len(grp) for grp in grouped.values()]) > 1:
            self._zday, self._zdaysum, self._zdaymax = {}, {}, {}
        # now create variables for different days and periods within each day
        for day, periods in grouped.iteritems():
            daystr = day.strftime(dsform)
            perstrs = [per.strftime(psform) for per in periods]
            if len(periods) > 1:
                self._zday[daystr] = self._model.addVar(
                    vtype=GRB.BINARY, name='zday[{}]'.format(daystr))
            for period, perstr in zip(periods, perstrs):
                self._red[perstr] = self._model.addVar(
                    vtype=GRB.CONTINUOUS, name='red[{}]'.format(perstr))
                self._z[perstr] = self._model.addVar(
                    vtype=GRB.BINARY, name='z[{}]'.format(perstr))
        # for the expMA we have to define a variable for the bl value
        # for every period of the simulation range
        for per in self._index:
            perstr = per.strftime(psform)
            self._bl[perstr] = self._model.addVar(
                vtype=GRB.CONTINUOUS, name='bl[{}]'.format(perstr))
        self._model.update()  # this must be done before defining constaints
        # determine "bigM" value from the bounds on the control variables
        M = np.sum(np.asarray(self._dynsys._opts['nrg_coeffs']) *
                   (self._dynsys._opts['umax'] - self._dynsys._opts['umin']),
                   axis=1).max()
        # if u is not bounded the the above results in an NaN value. We need
        # to deal with this in a better way than the following:
        if np.isnan(M):
            M = 1e9
        # perform some preparations for the constraints
        drcomp_ = []
        nrgcons = self._dynsys.get_consumption()['energy']
        lmps = LMP.tz_convert('US/Pacific').loc[locidx] / 1000  # to $/kWh
        holidays = USFederalHolidayCalendar().holidays(
            start=locidx.min(), end=locidx.max())
        isBusiness = (locidx.dayofweek < 5) & (~locidx.isin(holidays))
        isBusiness = pd.Series(isBusiness, index=locidx)
        # add constraints on varible zday (if multiple periods per day)
        for day, periods in grouped.iteritems():
            daystr = day.strftime(dsform)
            perstrs = [per.strftime(psform) for per in periods]
            if len(periods) > 1:
                self._zdaysum[daystr] = self._model.addConstr(
                    lhs=self._zday[daystr],
                    sense=GRB.LESS_EQUAL,
                    rhs=quicksum([self._z[ps] for ps in perstrs]),
                    name='zdaysum[{}]'.format(daystr))
                for period, perstr in zip(periods, perstrs):
                    self._zdaymax[perstr] = self._model.addConstr(
                        lhs=self._zday[daystr],
                        sense=GRB.GREATER_EQUAL,
                        rhs=self._z[perstr],
                        name='zdaymax[{}]'.format(perstr))
        self._model.update()
        # now add the constraints that define the baseline as well as a
        # bunch of other stuff
        for cons, alpha in zip([nrgcons[isBusiness], nrgcons[~isBusiness]],
                               [alpha_b, alpha_nb]):
            # localize consumption index
            considxloc = cons.index.tz_convert('US/Pacific')
            # compute BLs for each hour separately
            con_hrly = {hour: cons[considxloc.hour == hour].sort_index()
                        for hour in range(24)}
            for hour, con in con_hrly.items():
                # set the initial value of the BL to zero (this should not have
                # an overly large effect of the course of a year or so...)
                # NOTE: This assumes that the first occurrence of an hour (for
                # both business and non-business days) is NOT a potential event
                perstr_pre = con.index[0].strftime(psform)
                self._blcon[perstr_pre, 'init'] = self._model.addConstr(
                    lhs=self._bl[perstr_pre], sense=GRB.EQUAL,
                    rhs=0.0, name='blcon[{}]'.format(perstr_pre))
                # now loop through the rest
                for period, q in con.iloc[1:].iteritems():
                    perstr = period.strftime(psform)
                    # if the period under consideration is a DR period,
                    # we have to do some work ...
                    if period in valid_periods:
                        # need to use zday if this day has multiple DR events
                        dt = period.tz_convert('US/Pacific').date()
                        if len(grouped[dt]) > 1:
                            z = self._zday[dt.strftime(dsform)]
                        else:
                            z = self._z[perstr]
                        # add big M constraints on the bl
                        self._blcon[perstr, 'static'] = self._model.addConstr(
                            lhs=self._bl[perstr], sense=GRB.LESS_EQUAL,
                            rhs=self._bl[perstr_pre] + M * (1 - z),
                            name='blcon[{},static]'.format(perstr))
                        self._blcon[perstr, 'change'] = self._model.addConstr(
                            lhs=self._bl[perstr], sense=GRB.LESS_EQUAL,
                            rhs=alpha*q + (1-alpha)*self._bl[perstr_pre] + M*z,
                            name='blcon[{},change]'.format(perstr))
                        # add constraints on baseline reduction
                        self._redpos[perstr] = self._model.addConstr(
                            lhs=self._red[perstr], sense=GRB.GREATER_EQUAL,
                            rhs=0.0, name='redpos[{}]'.format(perstr))
                        self._redBL[perstr] = self._model.addConstr(
                            lhs=self._red[perstr], sense=GRB.LESS_EQUAL,
                            rhs=self._bl[perstr] - q,
                            name='redBL[{}]'.format(perstr))
                        self._red0[perstr] = self._model.addConstr(
                            lhs=self._red[perstr], sense=GRB.LESS_EQUAL,
                            rhs=self._z[perstr] * M,
                            name='red0[{}]'.format(perstr))
                        # add DR compensation to objective
                        drcomp_.append(
                            (lmps.loc[period.tz_convert('US/Pacific')] *
                             self._red[perstr]))
                    # ... otherwise this is pretty straightforward
                    else:
                        self._blcon[perstr] = self._model.addConstr(
                            lhs=self._bl[perstr], sense=GRB.EQUAL,
                            rhs=alpha * q + (1 - alpha) * self._bl[perstr_pre],
                            name='blcon[{}]'.format(perstr))
                    # update and keep track of last bl variable
                    perstr_pre = perstr
        drcomp = pd.Series(drcomp_, index=self._dr_periods)
        self._model.update()
        return drcomp

    def DR_comp_blfix(self, LMP, bl_values, **kwargs):
        """
            Return compensation for DR, i.e. reductions w.r.t. baseline.
            Here LMP is a pandas Series (indexed by a tz-aware pandas
            Datetimeindex containing all of the object's indices) and
            bl_values is a pandas Series, whose index is a DatetimeIndex,
            each entry of which represents a possible DR period, and whose
            values are the baseline values for those periods (assumed fixed).
            This is used for solving the baseline-taking equilibrium problem.
            Note that LMP may also be LMP-G, i.e. the LMP minus the generation
            component of the tariff.
        """

        self._removeOld()
        self._blvals = bl_values[
            bl_values.index.isin(self._index)].tz_convert('US/Pacific')
        locidx = self._index.tz_convert('US/Pacific')
        self._grouped = self._blvals.index.groupby(self._blvals.index.date)
        # define dictionaries to store variables in
        self._red, self._z = {}, {}
        self._redpos, self._redBL, self._red0 = {}, {}, {}
        # create variables for different days and periods within each day
        for day, periods in self._grouped.iteritems():
            perstrs = [per.strftime(psform) for per in periods]
            for period, perstr in zip(periods, perstrs):
                self._red[perstr] = self._model.addVar(
                    vtype=GRB.CONTINUOUS, name='red[{}]'.format(perstr))
                self._z[perstr] = self._model.addVar(
                    vtype=GRB.BINARY, name='z[{}]'.format(perstr))
        self._model.update()  # must be done before defining constaints
        # determine "bigM" value from the bounds on the control variables
        M = np.sum(np.asarray(self._dynsys._opts['nrg_coeffs']) *
                   (self._dynsys._opts['umax'] -
                    self._dynsys._opts['umin']), axis=1).max()
        # if u is not bounded the the above results in an NaN value. We
        # need to deal with this in a better way than the following:
        if np.isnan(M):
            M = 1e9
        # perform some preparations for the constraints
        self._drcomp = 0.0
        nrgcons = self._dynsys.get_consumption()['energy']
        DR_rewards = get_DR_rewards(LMP, isLMPmG=kwargs.get('isLMPmG'),
                                    tariff=kwargs.get('tariff'))
        # Pick out relevant dates and congvert to $/kWh
        DR_rewards = DR_rewards.tz_convert('US/Pacific').loc[locidx] / 1000
        holidays = USFederalHolidayCalendar().holidays(
            start=locidx.min(), end=locidx.max())
        isBusiness = (locidx.dayofweek < 5) & (~locidx.isin(holidays))
        isBusiness = pd.Series(isBusiness, index=locidx)
        # formulate constaints and add terms to objective
        for i, day in enumerate(self._grouped):
            periods = self._grouped[day]
            perstrs = [per.strftime(psform) for per in periods]
            for period, perstr in zip(periods, perstrs):
                # add constraints on baseline reduction
                self._redpos[perstr] = self._model.addConstr(
                    lhs=self._red[perstr], sense=GRB.GREATER_EQUAL,
                    rhs=0.0, name='redpos[{}]'.format(perstr))
                self._redBL[perstr] = self._model.addConstr(
                    lhs=(self._red[perstr] + nrgcons.loc[period] -
                         (1-self._z[perstr]) * M),
                    sense=GRB.LESS_EQUAL, rhs=self._blvals.loc[period],
                    name='redBL[{}]'.format(perstr))
                self._red0[perstr] = self._model.addConstr(
                    lhs=self._red[perstr], sense=GRB.LESS_EQUAL,
                    rhs=self._z[perstr] * M, name='red0[{}]'.format(
                        perstr))
                # add DR compensation to objective
                self._drcomp += DR_rewards.loc[period] * self._red[perstr]
        self._model.update()
        return self._drcomp

    def compute_baseline(self, bl_periods, red_times=None, BL='CAISO',
                         **kwargs):
        """
            Compute the CAISO baseline for all elements of the pandas
            Datetimeindex bl_periods. If red_times is a Datetimeindex,
            regard the associated days as "event days" (in addition to
            weekend days and holidays).
        """
        if BL == 'CAISO':
            return self._BL_CAISO(bl_periods, red_times=red_times)
        elif BL == 'expMA':
            return self._BL_expMA(bl_periods, red_times=red_times,
                                  **kwargs)
        else:
            raise NotImplementedError(
                'Baseline type "{}" not known!'.format(BL))

    def _BL_CAISO(self, bl_periods, red_times=None):
        """
            Compute the CAISO baseline for all elements of the pandas
            Datetimeindex bl_periods. If red_times is a Datetimeindex,
            regard the associated days as "event days" (in addition to
            weekend days and holidays).
        """
        locidx = self._index.tz_convert('US/Pacific')
        cons = self._dynsys.get_consumption()['energy'].tz_convert(
            'US/Pacific')
        holidays = USFederalHolidayCalendar().holidays(
            start=locidx.min(), end=locidx.max())
        isBusiness = (locidx.dayofweek < 5) & (~locidx.isin(holidays))
        isBusiness = pd.Series(isBusiness, index=locidx)
        if red_times is not None:
            isEventDay = locidx.normalize().isin(red_times.tz_convert(
                'US/Pacific').normalize())
        blidx, blvals = bl_periods.tz_convert('US/Pacific'), []
        for period in blidx:
            per_select = ((locidx < period) &
                          (locidx.hour == period.hour) &
                          (locidx.minute == period.minute))
            if isBusiness.loc[period]:
                nmax = 10
                per_select = per_select & isBusiness.values
            else:
                nmax = 4
                per_select = per_select & (~isBusiness.values)
            if red_times is not None:
                per_select = per_select & (~isEventDay)
            similars = locidx[per_select].sort_values(ascending=False)[:nmax]
            blvals.append(np.sum([c.getValue() for c in cons.loc[similars]]) /
                          float(len(similars)))
        return pd.Series(blvals, index=blidx.tz_convert('GMT'))

    def _BL_expMA(self, bl_periods, red_times=None, alpha_b=0.14,
                  alpha_nb=0.32):
        """
            Compute the expMA baseline for all elements of the pandas
            Datetimeindex bl_periods using the smoothing parameter alpha.
            If red_times is a Datetimeindex, regard the associated days as
            "event days" (in addition to weekend days and holidays).
        """
        locidx = self._index.tz_convert('US/Pacific')
        cons = self._dynsys.get_consumption()['energy'].tz_convert(
            'US/Pacific')
        cons = pd.Series([c.getValue() for c in cons],
                         index=cons.index)
        holidays = USFederalHolidayCalendar().holidays(
            start=locidx.min(), end=locidx.max())
        isBusiness = (locidx.dayofweek < 5) & (~locidx.isin(holidays))
        isBusiness = pd.Series(isBusiness, index=locidx)
        bls = []
        for con, alpha in zip([cons[isBusiness], cons[~isBusiness]],
                              [alpha_b, alpha_nb]):
            # determine intitial values for the BL from non-DR data
            if red_times is not None:
                nDRc = con[~con.index.isin(red_times)]
            else:
                nDRc = con
            cmeans = nDRc.groupby(nDRc.index.hour).mean()
            # compute BL for each hour separately
            con_hrly = {hour: con[con.index.hour == hour]
                        for hour in range(24)}
            bl_hrly = []
            for hour, conhr in con_hrly.items():
                blvals = [cmeans[hour]]
                if red_times is not None:
                    for period, c in conhr.iteritems():
                        if period in red_times:
                            blvals.append(blvals[-1])
                        else:
                            blvals.append(alpha*c + (1-alpha)*blvals[-1])
                else:
                    for period, c in conhr.iteritems():
                        blvals.append(alpha*c + (1-alpha)*blvals[-1])
                bl_hrly.append(pd.Series(blvals[1:], index=conhr.index))
            bls.append(pd.concat(bl_hrly).tz_convert('GMT'))
        return pd.concat(bls).loc[bl_periods]

    def optimize(self, tariff, LMP=None, dr_periods=None, BL='CAISO',
                 isRT=False, isPDP=False, carbon=False, **kwargs):
        """
            Solve the participant's optimization problem. Pass in additional
            Lin/Quad Expr of other objective terms with 'add_obj_term' kwarg
        """
        if isRT and (dr_periods is not None):
            raise Exception('Cannot combine DR with RTP.')
        if isPDP and (dr_periods is not None):
            raise Exception('Cannot combine DR with PDP.')
        # extract additonal objective term if given
        if 'add_obj_term' in kwargs:
            add_obj_term = kwargs['add_obj_term']
        else:
            add_obj_term = 0
        # energy charges are always included (demand charges
        # are set to zero if tariff has none and DR_compensation is
        # set to zero if there are no DR events ...)
        # if (LMP is None) or (dr_periods is None):
        #     #print drc
        #     drc = 0.0
        # else:
        #     #print self.DR_compensation(LMP, dr_periods, BL=BL,
        #     #                     tariff=tariff, **kwargs)
        #     drc=quicksum(self.DR_compensation(LMP, dr_periods, BL=BL,
        #                          tariff=tariff, **kwargs).values.tolist())
        self._model.setObjective(
            self._dynsys.additional_cost_term(vals=False) +
            quicksum(self.energy_charges(
                tariff, isRT=isRT, LMP=LMP, isPDP=isPDP,
                carbon=carbon).values) +
            quicksum(self.demand_charges(tariff, isPDP=False).values) -
            quicksum(self.DR_compensation(LMP, dr_periods, BL=BL,
                     tariff=tariff, **kwargs).values) +
            add_obj_term)
        self._model.optimize()

    def optimize_blfixed(self, tariff, LMP, bl_values, carbon=False, **kwargs):
        """
            Solve the participant's optimziation problem in case the baseline
            values are fixed.
        """
        # No option for RTPs. No biggie, since RTP and DR are alternatives.
        # extract additonal objective term if given
        if 'add_obj_term' in kwargs:
            add_obj_term = kwargs['add_obj_term']
        else:
            add_obj_term = 0
        self._model.setObjective(
            quicksum(self.energy_charges(tariff, LMP=LMP,
                                         carbon=carbon).values) +
            self._dynsys.additional_cost_term(vals=False))
        self._model.update()
        # for some tariffs we also have demand charges
        if tariff in dem_charges:
            self._model.setObjective(
                self._model.getObjective() +
                quicksum(self.demand_charges(tariff).values))
        else:
            if hasattr(self, '_maxcon'):
                for maxcon in self._maxcon.values():
                    self._model.remove(maxcon)
                del self._maxcon
            if hasattr(self, '_maxconbnd'):
                for maxconbnd in self._maxconbnd.values():
                    self._model.remove(maxconbnd)
                del self._maxconbnd
        self._model.update()
        self._nonDRobj = self._model.getObjective() + add_obj_term
        self._model.setObjective(
            self._nonDRobj - self.DR_comp_blfix(
                LMP, bl_values, tariff=tariff, **kwargs))
        self._model.optimize()

    def generation_cost(self, LMP, carbon=False):
        """
            Return the generation cost of the partipant's consumption (= price
            of consuption according to the LMPs) as a gurobipy LinExpr.
        """
        lmps = LMP.loc[self._index] / 1000  # select and convert price to $/kWh
        if carbon:
            lmps += pd.Series(carbon_costs).loc[self._index.tz_convert(
                'US/Pacific').year].values / 1000.0
        cons = self._dynsys.get_consumption()['energy']
        return quicksum([lmp * con for lmp, con in
                         zip(lmps.values, cons.values)])

    def get_results(self):
        """
            Return results of optimziation problem.
        """
        columns = {}
        xopt, uopt = self._dynsys.get_optvals()
        for i in range(xopt.shape[1]):
            columns['x{}'.format(i+1)] = xopt[:-1, i]
        for i in range(uopt.shape[1]):
            columns['u{}'.format(i+1)] = uopt[:, i]
        cons = self._dynsys.get_consumption()
        columns['nrg_cons'] = np.array([e.getValue() for e in cons['energy']])
        columns['pwr_cons'] = np.array([e.getValue() for e in cons['power']])
        dfs = [pd.DataFrame(columns, index=self._index)]
        if hasattr(self, '_z'):
            perstrs, vals = [], []
            for perstr, z in self._z.items():
                perstrs.append(perstr)
                vals.append(bool(z.X))
            dtidx = pd.to_datetime(perstrs, format=psform).tz_localize(
                'US/Pacific').tz_convert('GMT')
            dfs.append(pd.DataFrame({'z': vals}, index=dtidx))
        if hasattr(self, '_red'):
            perstrs, vals = [], []
            for perstr, red in self._red.items():
                perstrs.append(perstr)
                vals.append(red.X)
            dtidx = pd.to_datetime(perstrs, format=psform).tz_localize(
                'US/Pacific').tz_convert('GMT')
            dfs.append(pd.DataFrame({'red': vals}, index=dtidx))
        if hasattr(self, '_bl'):
            perstrs, vals = [], []
            for perstr, bl in self._bl.items():
                perstrs.append(perstr)
                vals.append(bl.X)
            dtidx = pd.to_datetime(perstrs, format=psform).tz_localize(
                'US/Pacific').tz_convert('GMT')
            dfs.append(pd.DataFrame({'BL': vals}, index=dtidx))
        return pd.concat(dfs, axis=1)

    def _removeOld(self):
        """
            Helper function removing all DR-related variables from the
            underlying gurobipy optimization model.
        """
        if hasattr(self, '_zday'):
            for zday in self._zday.values():
                self._model.remove(zday)
            del self._zday
        if hasattr(self, '_red'):
            for red in self._red.values():
                self._model.remove(red)
            del self._red
        if hasattr(self, '_z'):
            for z in self._z.values():
                self._model.remove(z)
            del self._z
        if hasattr(self, '_bl'):
            for bl in self._bl.values():
                self._model.remove(bl)
            del self._bl
        if hasattr(self, '_zdaysum'):
            for zdaysum in self._zdaysum.values():
                self._model.remove(zdaysum)
            del self._zdaysum
        if hasattr(self, '_zdaymax'):
            for zdaymax in self._zdaymax.values():
                self._model.remove(zdaymax)
            del self._zdaymax
        if hasattr(self, '_blcon'):
            for blcon in self._blcon.values():
                self._model.remove(blcon)
            del self._blcon
        if hasattr(self, '_redpos'):
            for redpos in self._redpos.values():
                self._model.remove(redpos)
            del self._redpos
        if hasattr(self, '_redBL'):
            for redBL in self._redBL.values():
                self._model.remove(redBL)
            del self._redBL
        if hasattr(self, '_red0'):
            for red0 in self._red0.values():
                self._model.remove(red0)
            del self._red0
        self._model.update()


def compute_BLtaking_eq(blmodel, tariff, LMP, dr_periods, BL='CAISO',
                        blinit='noDR', eps=0.005, maxiter=20, carbon=False,
                        **kwargs):
    """
        Function used ot compute Baseline-taking equilibrium.
    """
    if 'logger' in kwargs:
        logger = kwargs['logger']
        if 'isLMPmG' in kwargs:
            logstr = BL + ' (LMP-G)'
        else:
            logstr = BL
        logger.log(logging.INFO,
                   'Computing BL-taking eq. for ' '{} BL.'.format(logstr))
    dfs, blvals, objs, gencosts, residuals = [], [], [], [], []
    if blinit == 'gamed':
        blmodel.optimize(tariff, LMP=LMP, dr_periods=dr_periods,
                         BL=BL, carbon=carbon, **kwargs)
    elif blinit == 'noDR':
        blmodel.optimize(tariff, LMP=LMP, carbon=carbon, **kwargs)
    else:
        errmsg = 'Unknown BL initialization parameter {}.'.format(blinit)
        logger.log(logging.ERROR, errmsg)
        raise NotImplementedError(errmsg)
    # retrieve data from the solution for initialization
    dfs.append(blmodel.get_results())
    if 'red' in dfs[-1]:
        blvals.append(blmodel.compute_baseline(
            dr_periods, BL=BL, red_times=dfs[-1][dfs[-1]['red'] > 0].index))
    else:
        blvals.append(blmodel.compute_baseline(dr_periods, BL=BL))
    objs.append(blmodel._model.getObjective().getValue())
    gencosts.append(blmodel.generation_cost(LMP).getValue())
    residuals.append(np.NaN)
    # solve the bl-taking problem for the first time using the bl values
    # from the previous solution of the problem
    blmodel.optimize_blfixed(tariff, LMP=LMP, bl_values=blvals[-1],
                             carbon=carbon, **kwargs)
    dfs.append(blmodel.get_results())
    blvals.append(blmodel.compute_baseline(
            dr_periods, BL=BL, red_times=dfs[-1][dfs[-1]['red'] > 0].index))
    objs.append(blmodel._model.getObjective().getValue())
    gencosts.append(blmodel.generation_cost(LMP).getValue())
    residuals.append(np.linalg.norm(blvals[1] - blvals[0]))
    n_iter = 0
    while (residuals[-1] > eps) and (n_iter < maxiter):
        if 'logger' in kwargs:
            logger.log(logging.INFO,
                       'Residual: {:.2f}, '.format(residuals[-1]) +
                       'Continuing fixed point iteration.')
        blmodel.optimize_blfixed(
            tariff, LMP=LMP, bl_values=blvals[-1], carbon=carbon, **kwargs)
        dfs.append(blmodel.get_results())
        blvals.append(blmodel.compute_baseline(
                dr_periods, BL=BL,
                red_times=dfs[-1][dfs[-1]['red'] > 0].index))
        objs.append(blmodel._model.getObjective().getValue())
        gencosts.append(blmodel.generation_cost(LMP).getValue())
        residuals.append(np.linalg.norm(blvals[-2] - blvals[-1]))
        n_iter += 1
    if 'logger' in kwargs:
        if residuals[-1] <= eps:
            logger.log(logging.INFO,
                       'Fixed-point iteration successful. ' +
                       'BL-taking eq. found.')
        else:
            logger.log(logging.WARNING,
                       'Fixed-point iteration failed.' +
                       'No BL-taking eq. found. ')
    return dfs[-1]
