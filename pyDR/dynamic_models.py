"""
Code for the dynamical system component of the Baselining work.

@author: Maximilian Balandat
@date Jan 6, 2016
"""

import numpy as np
import pandas as pd
from scipy.signal import cont2discrete
from patsy import dmatrices
from gurobipy import quicksum, GRB, LinExpr


class DynamicalSystem(object):
    """
        Abstract base class for dynamical system optimization objects.
    """

    def __init__(self, model):
        """
            Construct an abstract dynamical system object based on the
            gurobipy Model object 'model'.
        """
        self._model = model

    def set_window(self, index):
        """
            Set the window for the optimization. Here index is a pandas
            DatetimeIndex.
        """
        self._index = index

    def get_consumption(self):
        """
            Returns power consumption (kW) and energy consumpiton (kWh) as
            gurobi LinExpr or QuadExpr
        """
        raise NotImplementedError('Function not implemented in base class.')

    def set_opts(self, **kwargs):
        """
            Set options (kwargs) to their respective values.
        """
        raise NotImplementedError('Function not implemented in base class.')

    def populate_model(self):
        """
            Add system dynamics and constraints as constraints to the gurobipy
            optimization model 'model'.
        """
        raise NotImplementedError('Function not implemented in base class.')


class LinearSystem(DynamicalSystem):
    """
        Linear Dynamical System class.
    """

    def __init__(self, model, A, B, **kwargs):
        """
            Constructor for a Linear Dynamical System. Here 'model' is a
            gurobipy Model object. The dynamics are
               x_{t+1} = A * x_t + B * u_t + E * v_t
                   y_t = C * x_t + D * u_t
            where x and u are vectors of the system state, respecively, and
            v is a vector of exogenous disturbances. If not specified, the
            matrices C defaults to the identity matrix and the matrices
            D and E default to zero matrix.
        """
        super(LinearSystem, self).__init__(model)
        self._opts = {}
        self._mats = {'A': np.asarray(A), 'B': np.asarray(B)}
        if 'C' in kwargs:
            self._mats['C'] = np.asarray(kwargs['C'])
        else:
            self._mats['C'] = np.eye(self._mats['A'].shape[1])
        self._dims = {'x': self._mats['A'].shape[1],
                      'u': self._mats['B'].shape[1],
                      'y': self._mats['C'].shape[0]}
        if 'D' in kwargs:
            self._mats['D'] = np.asarray(kwargs['D'])
        else:
            self._mats['D'] = np.zeros((self._dims['y'], self._dims['u']))
        if 'E' in kwargs:
            self._mats['E'] = np.asarray(kwargs['E'])
            self._dims['v'] = self._mats['E'].shape[1]
        if 'x0' in kwargs:
            self.x0 = kwargs['x0']
        self._changed = {'T': True}

    def set_opts(self, **kwargs):
        """
            Set options (kwargs) to their respective values.
            For example, to set the initial state 'x0' of the system to
            the value [2, 1], call self.set_opts(x0=[2, 1])
        """
        self._opts.update(kwargs)
        for kwarg in kwargs:
            if kwarg == 'x0':
                self.x0 = kwargs[kwarg]
            self._changed[kwarg] = True

    def set_window(self, index):
        """
            Set the window for the optimization. Here index is a pandas
            DatetimeIndex.
        """
        try:
            if not np.any(self._index == index):
                self._opts['T'] = len(index)
                self._changed['T'] = True
                self._index = index
        except (ValueError, AttributeError):
            self._opts['T'] = len(index)
            self._changed['T'] = True
            self._index = index

    def simulate(self, u, **kwargs):
        """
            Computes system evolution x and output y for the given control
            sequence u. If at least one of the matrices E or F has been
            specified, a disturbance sequence w of the same length must be
            provided as keyword argument.
        """
        T = u.shape[0]
        A, B, C = self._mats['A'], self._mats['B'], self._mats['C']
        hasD, hasE = ('D' in self._mats), ('E' in self._mats)
        x = np.zeros((T+1, self._dims['x']))
        y = np.zeros((T, self._dims['y']))
        x[0, :] = self._opts['x0']
        y[0, :] = np.inner(C, x[0, :])
        if hasD:
            D = self._mats['D']
            y[0, :] += np.inner(D, u[0, :])
        if hasE:
            E = self._mats['E']
            v = kwargs['v']
        for t in range(T):
            x[t+1, :] = np.inner(A, x[t, :]) + np.inner(B, u[t, :])
            if hasE:
                x[t+1, :] += np.inner(E, v[t, :])
            y[t, :] = np.inner(C, x[t, :])
            if hasD:
                y[t, :] += np.inner(D, u[t, :])
        # not sure what to do with the last time step, so leave it be for now
        # y[T, :] = np.inner(C, x[T, :])
        return x, y

    def populate_model(self, v=None, **kwargs):
        """
            Add system dynamics and constraints to the gurobipy optimization
            model. Overwrites any previous assignment.
        """
        T = self._opts['T']
        nu, nx = self._dims['u'], self._dims['x']
        if self._changed['T']:  # first time def or change of horizon
            # make sure to delete old optimization variables if they exist
            self._removeOld()
            # define optimization variables for input and state
            u, x = {}, {}
            for i in range(nu):
                for t in range(T):
                    u[t, i] = self._model.addVar(vtype=GRB.CONTINUOUS,
                                                 name='u[{},{}]'.format(t, i))
            for i in range(nx):
                for t in range(T+1):
                    x[t, i] = self._model.addVar(vtype=GRB.CONTINUOUS,
                                                 name='x[{},{}]'.format(t, i))
            # update the model so it knows the variables
            self._model.update()
            # add control constraints
            umin, umax = self._opts['umin'], self._opts['umax']
            has_umin, has_umax = ~np.isnan(umin), ~np.isnan(umax)
            for i in range(nu):
                for t in np.arange(has_umin.shape[0])[has_umin[:, i]]:
                    u[t, i].setAttr(GRB.Attr.LB, umin[t, i])
                for t in np.arange(has_umax.shape[0])[has_umax[:, i]]:
                    u[t, i].setAttr(GRB.Attr.UB, umax[t, i])
            # update intitial state, if provided
            if 'x0' in kwargs:
                self.x0 = kwargs['x0']
            # add constraint on initial state
            self.x0con = {}
            for i in range(nx):
                self.x0con[0, i] = self._model.addConstr(
                    lhs=x[0, i], sense=GRB.EQUAL, rhs=self.x0[i],
                    name='dyn[0,{}]'.format(i))
            # add system dynamics
            A, B = self._mats['A'], self._mats['B']
            if ('E' in self._mats):
                w = np.inner(v, self._mats['E'])
            else:
                w = np.zeros((T, nx))
            # dynamic evolution of state and output
            self.dyncon = {}
            for t in range(T):
                for i in range(nx):
                    # put w on RHS to speed up constraint updates
                    self.dyncon[t, i] = self._model.addConstr(
                        lhs=(x[t+1, i] - quicksum([A[i, k] * x[t, k]
                                                   for k in range(nx)]) -
                             quicksum([B[i, k] * u[t, k] for k in range(nu)])),
                        sense=GRB.EQUAL, rhs=w[t, i],
                        name='dyn[{},{}]'.format(t+1, i))
            self._model.update()
            # add state constraints
            xmin, xmax = self._opts['xmin'], self._opts['xmax']
            has_xmin, has_xmax = ~np.isnan(xmin), ~np.isnan(xmax)
            for i in range(nx):
                for t in np.arange(has_xmin.shape[0])[has_xmin[:, i]]:
                    x[t+1, i].setAttr(GRB.Attr.LB, xmin[t, i])
                for t in np.arange(has_xmax.shape[0])[has_xmax[:, i]]:
                    x[t+1, i].setAttr(GRB.Attr.UB, xmax[t, i])
            self._model.update()
            # indicate that model is up to date
            for name in ['T', 'x0', 'umin', 'umax', 'xmin', 'xmax', 'v']:
                self._changed[name] = False
            # make variables accessible as object variables
            self.u, self.x, self.v = u, x, v
        else:
            # change input constraints
            if self._changed['umin']:
                umin = self._opts['umin']
                for i in range(nu):
                    for t in range(T):
                        self.u[t, i].setAttr(GRB.Attr.LB, umin[t, i])
                self._changed['umin'] = False
            if self._changed['umax']:
                umax = self._opts['umax']
                for i in range(nu):
                    for t in range(T):
                        self.u[t, i].setAttr(GRB.Attr.UB, umax[t, i])
                self._changed['umax'] = False
            # change state constraints
            if self._changed['xmin']:
                xmin = self._opts['xmin']
                # xmin[np.isnan(xmin)] = - np.Inf
                for i in range(nx):
                    for t in range(T):
                        self.x[t+1, i].setAttr(GRB.Attr.LB, xmin[t, i])
                self._changed['xmin'] = False
            if self._changed['xmax']:
                xmax = self._opts['xmax']
                # xmax[np.isnan(xmax)] = np.Inf
                for i in range(nx):
                    for t in range(T):
                        self.x[t+1, i].setAttr(GRB.Attr.UB, xmax[t, i])
                self._changed['xmax'] = False
            # change initial state
            if self._changed['x0']:
                for i in range(nx):
                    self._model.getConstrByName('dyn[0,{}]'.format(i)).setAttr(
                        GRB.Attr.RHS, self.x0[i])
                self._changed['x0'] = False
            # change effect of disturbance vector on dynamics (if any)
            if v is not None:
                if not np.all(v == self.v):
                    self.v = v
                    w = np.inner(v, self._mats['E'])
                    for i in range(nx):
                        for t in range(T):
                            self._model.getConstrByName(
                                'dyn[{},{}]'.format(t+1, i)).setAttr(
                                GRB.Attr.RHS, w[t, i])
            # finally update and include all changes
            self._model.update()

    def _removeOld(self):
        """
            Helper function removing all optimization variables from the
            underlying gurobipy optimization model if the time horizon
            has been changed.
        """
        if hasattr(self, 'u'):
            for u in self.u.values():
                self._model.remove(u)
            del self.u
        if hasattr(self, 'x'):
            for x in self.x.values():
                self._model.remove(x)
            del self.x
        if hasattr(self, 'x0con'):
            for x0c in self.x0con.values():
                self._model.remove(x0c)
            del self.x0con
        if hasattr(self, 'dyncon'):
            for dc in self.dyncon.values():
                self._model.remove(dc)
            del self.dyncon
        self._model.update()

    def get_optvals(self):
        """
            Return the optimal values of state and control.
        """
        xopt = np.array([[self.x[t, i].X for i in range(self._dims['x'])]
                        for t in range(self._opts['T']+1)])
        uopt = np.array([[self.u[t, i].X for i in range(self._dims['u'])]
                        for t in range(self._opts['T'])])
        return xopt, uopt

    def get_consumption(self):
        """
            Return power consumption (in kW) and energy consumption (in kWh)
            of the system during each optimization time interval as a pandas
            Dataframe of gurobipy LinExpr or QuadExpr.
        """
        if self._changed['T'] | self._changed['nrg_coeffs']:
            nrgcoeffs = np.asarray(self._opts['nrg_coeffs'])
            nu = self._dims['u']
            pwrcons = [LinExpr(nrgcoeffs, [self.u[t, i] for i in range(nu)])
                       for t in range(self._opts['T'])]
            # rescale to kWh (if time index resolution is different from 1h)
            rescale = self._index.freq.delta.total_seconds() / 3600.0
            nrgcons = [pc * rescale for pc in pwrcons]
            self._cons = pd.DataFrame({'power': pwrcons, 'energy': nrgcons},
                                      index=self._index)
        return self._cons

    def additional_cost_term(self, **kwargs):
        """
            Returns an additional cost term (a Gurobipy LinExp or QuadExp) that
            is included into the optimization problem. In the base class this
            is just zero.
        """
        return 0.0


class FraukesModel(LinearSystem):
    """
        The linear model of the Swiss building as used by Frauke
        in her paper.
    """

    def __init__(self, model, ts=15):
        """
            Create an instance of Frauke's Model with sampling time
            ts (in minutes)
        """
        from .utils import matrices_frauke
        A, B, E = matrices_frauke(ts)
        super(FraukesModel, self).__init__(model, A, B, E=E)

    def set_v(self, df):
        """
            Set model data (outside temperature, solar radiation, occupancy).
            Here df is a pandas dataframe indexed by a (timezonez-aware)
            Datetimeindex with columns outside_temp, solar_rad and occupancy
        """
        self.v = df
        v = df[['outside_temp', 'solar_rad', 'occupancy']].values
        self.set_opts(T=v.shape[0])
        self.populate_model(self, v=v)


class GenericBufferedProduction(LinearSystem):
    """
        A linear model for a generic buffered production model. Here it is
        assumed that all production costs are sunk costs except for energy.
        There is a battery with charging and discharging inefficiencies,
        as well as leakage, which can be used to store energy.
    """

    def __init__(self, model, eta_c=0.9, eta_d=0.9, Tleak=96, ts=15):
        """
            Create an instance of a generic buffered production model. Here
            eta_c and eta_d are the charging and discharging efficiencies of
            the battery, respectively, Tleak is time constant (in hours) of
            the charge leakage, and ts is the sampling time (in minutes).
        """
        Act = np.array([[-1.0/(Tleak*3600), 0], [0, 0]])
        Bct = np.array([[eta_c, -1.0/eta_d], [0, 1]])
        Cct = np.array([[0, 0]])
        Dct = np.array([[0, 0]])
        (A, B, C, D, dt) = cont2discrete(
            (Act, Bct, Cct, Dct), ts*60, method='zoh')
        super(GenericBufferedProduction, self).__init__(model, A, B)


class QuadraticUtility(DynamicalSystem):
    """
        A static quadratic consumption utiltiy model with time-seperable
        utilities. Used to benchmark the QuadraticUtilityWithBattery model.
    """

    def __init__(self, model, **kwargs):
        """
            Constructor for QuadraticUtility model. Here 'model' is a
            gurobipy Model object.
        """
        super(QuadraticUtility, self).__init__(model)
        self._opts = {}
        self._dims = {'u': 1}
        self._changed = {'T': True}

    def set_opts(self, **kwargs):
        """
            Set options (kwargs) to their respective values.
        """
        self._opts.update(kwargs)
        for kwarg in kwargs:
            self._changed[kwarg] = True

    def set_window(self, index):
        """
            Set the window for the optimization. Here index is a pandas
            DatetimeIndex.
        """
        try:
            if not np.any(self._index == index):
                self._opts['T'] = len(index)
                self._changed['T'] = True
                self._index = index
        except (ValueError, AttributeError):
            self._opts['T'] = len(index)
            self._changed['T'] = True
            self._index = index

    def compute_util_params(self, load_shape, nrg_charges, eta=-0.1,
                            fit='saturated'):
        """
            Computes the parameters of the quadratic utility function by
            calibrating the consumption under the energy charges nrg_charges
            to the provided load shape. If fit=='saturated', then compute a
            single parameter for each period. If fit=='regression', use OLS
            to estimate a model with a single parameter for each interaction
            of month, weekday/weekend and hour of day.
        """
        if fit == 'saturated':
            # Series of linear coefficients
            self._alpha = nrg_charges.loc[self._index] * (1 - 1/eta)
            # Series of quadratic coefficients
            self._beta = -1.0*nrg_charges.loc[self._index].divide(
                load_shape.loc[self._index]*eta, axis='index')
        elif fit == 'regression':
            df = pd.concat([nrg_charges, load_shape], axis=1)
            df = df.rename(columns={nrg_charges.name: 'p',
                                    load_shape.name: 'q'})
            df['month'] = df.index.month
            df['HoD'] = df.index.hour
            df['wknd'] = (df.index.dayofweek >= 5).astype(int)
            df['a_indiv'] = df['p'] * (1 - 1 / eta)
            df['b_indiv'] = -1.0 * df['p'] / (df['q'] * eta)
            y_a, X_a = dmatrices('a_indiv ~ -1 + C(month):C(HoD):C(wknd)',
                                 data=df.loc[self._index])
            y_b, X_b = dmatrices('b_indiv ~ -1 + C(month):C(HoD):C(wknd)',
                                 data=df.loc[self._index])
            # Note: the following weird syntax is necessary to convert patsy
            # DesignMatrix objects to np arrays - o/w this creates issues
            # when using multiprocessing since DesignMatrix objects cannot
            # be pickled (hopefully to be fixed in a later patsy version)
            _alpha = np.dot(np.asarray(X_a), np.linalg.lstsq(
                np.asarray(X_a), np.asarray(y_a).flatten())[0])
            self._alpha = pd.Series(_alpha, index=self._index)
            _beta = np.dot(np.asarray(X_b), np.linalg.lstsq(
                np.asarray(X_b), np.asarray(y_b).flatten())[0])
            self._beta = pd.Series(_beta, index=self._index)
        else:
            raise Exception('Unknown value for parameter "fit".')

    def get_optvals(self):
        """
            Return the optimal values of state and control.
        """
        xopt = np.array([[] for t in range(self._opts['T']+1)])
        uopt = np.array([[self.u[t, i].X for i in range(self._dims['u'])]
                        for t in range(self._opts['T'])])
        return xopt, uopt

    def get_indiv_us(self, vals=False):
        """
            Returns a dataframe containing the individual inputs. If vals is
            True, return the value (requires that the optimization problem was
            solved previously), otherwise return Gurobipy variables.
        """
        if vals:
            return pd.DataFrame(
                {i: [self.u[t, i].X for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)
        else:
            return pd.DataFrame(
                {i: [self.u[t, i] for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)

    def get_consumption(self):
        """
            Return power consumption (in kW) and energy consumption (in kWh)
            of the system during each optimization time interval as a pandas
            Dataframe of gurobipy LinExpr or QuadExpr.
        """
        if self._changed['T'] | self._changed['nrg_coeffs']:
            nrgcoeffs = np.asarray(self._opts['nrg_coeffs'])
            nu = self._dims['u']
            pwrcons = [LinExpr(nrgcoeffs, [self.u[t, i] for i in range(nu)])
                       for t in range(self._opts['T'])]
            # rescale to kWh (if time index resolution is different from 1h)
            rescale = self._index.freq.delta.total_seconds() / 3600.0
            nrgcons = [pc * rescale for pc in pwrcons]
            self._cons = pd.DataFrame({'power': pwrcons, 'energy': nrgcons},
                                      index=self._index)
        return self._cons

    def get_total_consumptions(self, vals=False):
        """
            Returns a Series with the total consumptions in each period. If
            vals is True, return the values (requires that the optimization
            problem was solved previously), otherwise return Gurobipy LinExps.
        """
        if vals:
            return pd.Series(
                [self.u[t, 0].X for t in range(self._opts['T'])],
                index=self._index)
        else:
            return pd.Series(
                [self.u[t, 0] for t in range(self._opts['T'])],
                index=self._index)

    def get_consumption_utilities(self, vals=False):
        """
            Returns a Series of consumption utilities.
        """
        tot_cons = self.get_total_consumptions(vals=vals)
        return self._alpha.loc[self._index].multiply(
            tot_cons, axis='index').add(
            -0.5*self._beta.loc[self._index].multiply(
                tot_cons**2, axis='index'))

    def additional_cost_term(self, vals=False):
        """
            Returns the (negative) quadratic utility as a Gurobipy QuadExp),
            the parameters alpha and beta of which were previously calibrated.
        """
        if vals:
            return self.get_consumption_utilities(vals=True).values.sum()
        else:
            return -quicksum([util for util in
                              self.get_consumption_utilities(
                                vals=False).values.flatten()])

    def populate_model(self, **kwargs):
        """
            Add system dynamics and constraints to the gurobipy optimization
            model. Overwrites any previous assignment.
        """
        T = self._opts['T']
        nu = self._dims['u']
        if self._changed['T']:  # first time def or change of horizon
            # make sure to delete old optimization variables if they exist
            self._removeOld()
            # define optimization variables for input
            u = {}
            for i in range(nu):
                for t in range(T):
                    u[t, i] = self._model.addVar(vtype=GRB.CONTINUOUS,
                                                 name='u[{},{}]'.format(t, i))
            # update the model so it knows the variables
            self._model.update()
            # add control constraints
            umin, umax = self._opts['umin'], self._opts['umax']
            has_umin, has_umax = ~np.isnan(umin), ~np.isnan(umax)
            for i in range(nu):
                for t in np.arange(has_umin.shape[0])[has_umin[:, i]]:
                    u[t, i].setAttr(GRB.Attr.LB, umin[t, i])
                for t in np.arange(has_umax.shape[0])[has_umax[:, i]]:
                    u[t, i].setAttr(GRB.Attr.UB, umax[t, i])
            self._model.update()
            # indicate that model is up to date
            for name in ['T', 'umin', 'umax']:
                self._changed[name] = False
            # make variables accessible as object variables
            self.u = u
        else:
            # change input constraints
            if self._changed['umin']:
                umin = self._opts['umin']
                for i in range(nu):
                    for t in range(T):
                        self.u[t, i].setAttr(GRB.Attr.LB, umin[t, i])
                self._changed['umin'] = False
            if self._changed['umax']:
                umax = self._opts['umax']
                for i in range(nu):
                    for t in range(T):
                        self.u[t, i].setAttr(GRB.Attr.UB, umax[t, i])
                self._changed['umax'] = False
            # finally update and include all changes
            self._model.update()

    def _removeOld(self):
        """
            Helper function removing all optimization variables from the
            underlying gurobipy optimization model if the time horizon
            has been changed.
        """
        if hasattr(self, 'u'):
            for u in self.u.values():
                self._model.remove(u)
            del self.u


class QuadraticUtilityWithBattery(LinearSystem):
    """
        A dynamic model for a quadratic consumption uitility with an additional
        battery to allow for storage and hence inter-temporal substitution of
        energy consumption. The battery has charging and discharging
        inefficiencies, as well as leakage.
    """

    def __init__(self, model, eta_c=0.95, eta_d=0.95, Tleak=96, ts=15):
        """
            Create an instance of a QuadraticUtilityWithBattery model. Here
            eta_c and eta_d are the battery's charging and discharging
            efficiencies, respectively, Tleak is time constant (in hours) of
            the charge leakage, and ts is the sampling time (in minutes).
        """
        Act = np.array([[-1.0/(Tleak*3600)]])
        # Act = np.expand_dims(Act, axis=0)
        Bct = np.array([[eta_c, -1.0/eta_d, 0]])
        Cct = np.array([[0]])
        Dct = np.array([[0, 0, 0]])
        (A, B, C, D, dt) = cont2discrete(
            (Act, Bct, Cct, Dct), ts*60, method='zoh')
        B = np.array([[eta_c, -1.0/eta_d, 0]])
        super(QuadraticUtilityWithBattery, self).__init__(model, A, B)

    def compute_util_params(self, load_shape, nrg_charges, eta=-0.1,
                            fit='saturated'):
        """
            Computes the parameters of the quadratic utility function by
            calibrating the consumption under the energy charges nrg_charges
            to the provided load shape. If fit=='saturated', then compute a
            single parameter for each period. If fit=='regression', use OLS
            to estimate a model with a single parameter for each interaction
            of month, weekday/weekend and hour of day.
        """
        if fit == 'saturated':
            # Series of linear coefficients
            self._alpha = nrg_charges.loc[self._index] * (1 - 1/eta)
            # Series of quadratic coefficients
            self._beta = -1.0*nrg_charges.loc[self._index].divide(
                load_shape.loc[self._index]*eta, axis='index')
        elif fit == 'regression':
            df = pd.concat([nrg_charges, load_shape], axis=1)
            df = df.rename(columns={nrg_charges.name: 'p',
                                    load_shape.name: 'q'})
            df['month'] = df.index.month
            df['HoD'] = df.index.hour
            df['wknd'] = (df.index.dayofweek >= 5).astype(int)
            df['a_indiv'] = df['p'] * (1 - 1 / eta)
            df['b_indiv'] = -1.0 * df['p'] / (df['q'] * eta)
            y_a, X_a = dmatrices('a_indiv ~ -1 + C(month):C(HoD):C(wknd)',
                                 data=df.loc[self._index])
            y_b, X_b = dmatrices('b_indiv ~ -1 + C(month):C(HoD):C(wknd)',
                                 data=df.loc[self._index])
            # Note: the following weird syntax is necessary to convert patsy
            # DesignMatrix objects to np arrays - o/w this creates issues
            # when using multiprocessing since DesignMatrix objects cannot
            # be pickled (hopefully to be fixed in a later patsy version)
            _alpha = np.dot(np.asarray(X_a), np.linalg.lstsq(
                np.asarray(X_a), np.asarray(y_a).flatten())[0])
            self._alpha = pd.Series(_alpha, index=self._index)
            _beta = np.dot(np.asarray(X_b), np.linalg.lstsq(
                np.asarray(X_b), np.asarray(y_b).flatten())[0])
            self._beta = pd.Series(_beta, index=self._index)
        else:
            raise Exception('Unknown value for parameter "fit".')

    def get_indiv_us(self, vals=False):
        """
            Returns a dataframe containing the individual inputs. If vals is
            True, return the value (requires that the optimization problem was
            solved previously), otherwise return Gurobipy variables.
        """
        if vals:
            return pd.DataFrame(
                {i: [self.u[t, i].X for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)
        else:
            return pd.DataFrame(
                {i: [self.u[t, i] for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)

    def get_total_consumptions(self, vals=False):
        """
            Returns a Series with the total consumptions in each period. If
            vals is True, return the values (requires that the optimization
            problem was solved previously), otherwise return Gurobipy LinExps.
        """
        if vals:
            return pd.Series([self.u[t, 1].X + self.u[t, 2].X
                              for t in range(self._opts['T'])],
                             index=self._index)
        else:
            return pd.Series([self.u[t, 1] + self.u[t, 2]
                              for t in range(self._opts['T'])],
                             index=self._index)

    def get_consumption_utilities(self, vals=False):
        """
            Returns a Series of consumption utilities.
        """
        tot_cons = self.get_total_consumptions(vals=vals)
        return self._alpha.loc[self._index].multiply(
            tot_cons, axis='index').add(
            -0.5*self._beta.loc[self._index].multiply(
                tot_cons**2, axis='index'))

    def additional_cost_term(self, vals=False):
        """
            Returns the (negative) quadratic utility as a Gurobipy QuadExp),
            the parameters alpha and beta of which were previously calibrated.
        """
        if vals:
            return self.get_consumption_utilities(vals=True).values.sum()
        else:
            return -quicksum([util for util in
                              self.get_consumption_utilities(
                                vals=False).values.flatten()])


class QuadUtilPerfectSub(LinearSystem):
    """
        A dynamic model for a utiltiy function that is quadatic in the total
        amount consumed (irrespective of when it is consumed), with an
        additional battery to allow for storage. This can be seen as a
        benchmark case for cost-free inter-temporal substitution.The battery
        has charging and discharging inefficiencies, as well as leakage.
    """

    def __init__(self, model, eta_c=0.95, eta_d=0.95, Tleak=96, ts=15):
        """
            Create an instance of a QuadUtilPerfectSub model. Here a and b are
            the parameters of the utility function u(q)=a*q-0.5*b*q^2, eta_c
            and eta_d are the battery's charging and discharging efficiencies,
            respectively, Tleak is time constant (in hours) of the charge
            leakage, and ts is the sampling time (in minutes).
        """
        Act = np.array([[-1.0/(Tleak*3600)]])
        # Act = np.expand_dims(Act, axis=0)
        Bct = np.array([[eta_c, -1.0/eta_d, 0]])
        Cct = np.array([[0]])
        Dct = np.array([[0, 0, 0]])
        (A, B, C, D, dt) = cont2discrete(
            (Act, Bct, Cct, Dct), ts*60, method='zoh')
        B = np.array([[eta_c, -1.0/eta_d, 0]])
        super(QuadUtilPerfectSub, self).__init__(model, A, B)

    def compute_util_params(self, load_shape, nrg_charges, eta=-0.1):
        """
            Computes the parameters of the quadratic utility function by
            calibrating the consumption under the energy charges nrg_charges
            to the provided load shape.
        """
        q = load_shape.loc[self._index].sum()
        p = nrg_charges.loc[self._index].values.mean()
        self._a = -p*(1-eta)/eta
        self._b = (self._a-p)/q

    def get_indiv_us(self, vals=False):
        """
            Returns a dataframe containing the individual inputs. If vals is
            True, return the value (requires that the optimization problem was
            solved previously), otherwise return Gurobipy variables.
        """
        if vals:
            return pd.DataFrame(
                {i: [self.u[t, i].X for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)
        else:
            return pd.DataFrame(
                {i: [self.u[t, i] for t in range(self._opts['T'])]
                    for i in range(self._dims['u'])},
                index=self._index)

    def get_total_consumptions(self, vals=False):
        """
            Returns a Series with the total consumptions in each period. If
            vals is True, return the values (requires that the optimization
            problem was solved previously), otherwise return Gurobipy LinExps.
        """
        if vals:
            return pd.Series([self.u[t, 1].X + self.u[t, 2].X
                              for t in range(self._opts['T'])],
                             index=self._index)
        else:
            return pd.Series([self.u[t, 1] + self.u[t, 2]
                              for t in range(self._opts['T'])],
                             index=self._index)

    def populate_model(self, **kwargs):
        """
            Add system dynamics and constraints to the gurobipy optimization
            model. Overwrites any previous assignment.
        """
        super(QuadUtilPerfectSub, self).populate_model()
        if not hasattr(self, 'q'):
            self.q = self._model.addVar(vtype=GRB.CONTINUOUS, name='q')
            self._model.update()
            q = quicksum(self.get_total_consumptions(vals=False).values)
            self.q_cons = self._model.addConstr(
                lhs=self.q, sense=GRB.EQUAL, rhs=q, name='q_cons')
            self._model.update()

    def additional_cost_term(self, vals=False):
        """
            Returns the (negative) quadratic utility on the total consumption.
        """
        if vals:
            q = self.q.X
        else:
            q = self.q
        return -self._a*q + 0.5*self._b*q*q
