"""
Functions for interfacing a GP model from a FoKL class with a Pyomo model.

| Function      | Description                                                                      |
|---------------|----------------------------------------------------------------------------------|
| fix_betas     | set values of the GP's beta coefficients defined as variables in the Pyomo model |
| fokl_to_pyomo | embed GP from FoKL model into Pyomo model                                        |

"""
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.environ import *
import warnings


def fix_betas(m, betas, i=0, scenarios=None):
    """
    Fix the already-initialized Pyomo beta Param's to scalar values.
    
    | Argument  | Type    | Description                         |
    |-----------|---------|-------------------------------------|
    | m         | -       | see 'fokl_to_pyomo'                 |
    | betas     | ndarray | shape is (draws, terms) or (terms,) |
    | i         | int     | index of next available GP in 'm'   |
    | scenarios | -       | see 'fokl_to_pyomo'                 |

    """
    beta = m.component(f"GP{i}_beta")
    terms = m.component(f"GP{i}_terms")
    
    if betas.ndim == 1:  # format (n,) to (1,n)
        betas = betas[np.newaxis, :]

    _INDEX_S = False
    if scenarios is not None:  # check this in case 'len(scenarios) == 1'
        _INDEX_S = True
        
    if _INDEX_S is False and len(beta.index_set()) == betas.shape[1]:  # then 'beta[term]' == average of 'betas', assuming 'len(scenarios) != 1'
        if scenarios is not None:
            warnings.warn("Ignoring 'scenarios'.", category=UserWarning)
        beta_avg = np.mean(betas, axis=0)
        for term in terms:
            beta[term] = beta_avg[term]

    else:  # then 'beta[s, term]' == 'betas[-(s + 1), term]'; i.e., most recent draws
        if scenarios is None:
            raise ValueError("'scenarios' must be passed to 'fix_betas' if used to index 'm.GP#_beta'. Otherwise, ensure 'betas' aligns with 'm.GP#_beta'.")
        s_ind = -1  # defined as separate index since 'scenarios' is not necessarily 'range(n)' but could be strings, etc.
        for s in scenarios:
            s_ind += 1
            for term in terms:
                beta[s, term] = betas[-(s_ind + 1), term]

    return m


def fokl_to_pyomo(self, xvars, yvar, m=None, t=None, scenarios=None):
    """
    Convert GP model from FoKL class to Pyomo model.
    
    | Argument  | Type                    | Description |
    |-----------|-------------------------|-------------|
    | xvars     | str, Var, DerivativeVar |             |
    | yvar      | str, Var, DerivativeVar |             |
    | m         | ConcreteModel           |             |
    | t         | ContinuousSet           |             |
    | scenarios | Set                     |             |

    """
    # Pre-processing:
    i = 0  # index of next available GP (in case Pyomo model contains multiple GPs)
    if m is None:
        m = pyo.ConcreteModel("FoKL-to-Pyomo Model")
    else:
        while m.find_component(f"GP{i}") is not None:
            i += 1
    s = scenarios  # rename
    
    # Check for type errors:
    if not isinstance(xvars, list):
        raise TypeError()
    elif len(xvars) != self.mtx.shape[1]:
        raise ValueError("'xvars' must have an element for each FoKL model training input.")
    _CREATE_VARS = False
    for var in xvars + [yvar]:
        if not any(isinstance(var, type) for type in [str, pyo.Var, dae.DerivativeVar]):
            raise TypeError()
        elif isinstance(var, str):
            _CREATE_VARS = True  # str received, so need to create Var
    if not isinstance(m, pyo.ConcreteModel):
        raise TypeError()
    if not (t is None or isinstance(t, dae.ContinuousSet)):
        raise TypeError()
    if not (s is None or isinstance(s, pyo.Set)):
        raise TypeError()

    # Pre-processing (cont.):

    jj = len(xvars)
    m.add_component(f"GP{i}_attributes", pyo.Set(initialize=range(jj)))   # indices of input variables
    attributes = m.component(f"GP{i}_attributes")

    if t is None and s is None:
        ts = None  # Pyomo indices
    elif t is None:
        ts = [s]
    elif s is None:
        ts = [t]
    else:
        ts = [t, s]
    
    if _CREATE_VARS:  # create Var for str in [xvars, yvar]
        for j in attributes:
            if isinstance(xvars[j], str):
                if ts is None:
                    xvars[j] = pyo.Var(bounds=self.minmax[j])
                else:
                    if len(ts) == 1:
                        xvars[j] = pyo.Var(ts[0], bounds=self.minmax[j])
                    else:
                        xvars[j] = pyo.Var(ts[0], ts[1], bounds=self.minmax[j])
        if isinstance(yvar, str):
            if ts is None:
                yvar = pyo.Var()
            else:
                if len(ts) == 1:
                    yvar = pyo.Var(ts[0])
                else:
                    yvar = pyo.Var(ts[0], ts[1])
    vars = xvars + [yvar]

    dims = 0  # number of dimensions; i.e., 1 if 't' or 'scenarios', 2 if both
    if ts is None:
        ts = [None] * (jj + 1)  # +1 to include yvar
        switch = np.zeros_like(ts, dtype=int)
    else:
        ts = []
        switch = []  # switch case; 0 for None, 1 for s, 2 for t, 3 for [t, s]

        def _index_error(_index):
            return ValueError(f"Length of 'xvars[{j}]' failed to match length of '{_index}'. Ensure the variable is not indexed by another set, and remove all indices if 'xvars[{j}]' should not be indexed by '{_index}'.")

        lt, ls = 1, 1
        if t is not None:
            lt = len(t)
            dims += 1
        if s is not None:
            ls = len(s)
            dims += 1
        if dims == 2:
            if lt == ls:
                raise ValueError("Lengths of 't' and 'scenarios' must be unique.")
            ltxls = int(lt * ls)

        for j in range(jj + 1):
            nd = vars[j].dim()  # number of dimensions, i.e., of indices
            lv = len(vars[j].index_set())  # length of variable

            if nd > dims:
                raise NotImplementedError("'xvars' and 'yvar' indexed by more than 't' and/or 'scenarios' is not supported.")
            elif nd == 0:
                ts.append(None)
                switch.append(0)
            elif nd == 1:
                if dims == 1:
                    if t is None:
                        if lv != ls:
                            raise _index_error("scenarios")
                        ts.append([s])
                        switch.append(1)
                    elif s is None:
                        if lv != lt:
                            raise _index_error("t")
                        ts.append([t])
                        switch.append(2)
                else:
                    if lv == ls:
                        ts.append([s])
                        switch.append(1)
                    elif lv == lt:
                        ts.append([t])
                        switch.append(2)
                    else:
                        raise ValueError(f"Index of 'xvars[{j}]' failed to be inferred from length because it is not equal to length of 't' nor 'scenarios'.")
            elif nd == 2:
                if lv != ltxls:
                    raise _index_error("[t, scenarios]")
                ts.append([t, s])
                switch.append(3)
    
    if t is not None and not any(switch[j] == ft for ft in [2, 3] for j in range(lv)):  # user passed 't' but no variables are indexed by 't'
        warnings.warn("Ignoring 't' as no 'xvars' nor 'yvar' appear to be indexed by 't'.", category=UserWarning)
        t = None

    # Some constants:
    mtx = np.array(self.mtx, dtype=int)  # indices/orders of basis functions (where 1 is B1 and 0 means none)

    # Some sets:
    m.add_component(f"GP{i}_terms", pyo.Set(initialize = range(mtx.shape[0] + 1)))    # terms (including beta0)
    terms = m.component(f"GP{i}_terms")

    # Define beta coefficients:
    if s is not None:  # index betas by scenarios and terms
        m.add_component(f"GP{i}_beta", pyo.Param(s, terms, mutable=True))  # 'mutable=True', to change the values dynamically
    else:  # index betas by terms only
        m.add_component(f"GP{i}_beta", pyo.Param(terms, mutable=True))
    fix_betas(m, self.betas, i, s)
    beta = m.component(f"GP{i}_beta")

    # Normalize attributes:

    def _normalize(x, minmax):
        """Normalization constraint."""
        return (x - minmax[0]) / (minmax[1] - minmax[0])

    def _eq_norm_00(m):
        return _normalize(xvars[j], self.minmax[j])

    def _eq_norm_01(m, s_ind):
        return _normalize(xvars[j][s_ind], self.minmax[j])

    def _eq_norm_10(m, t_ind):
        return _normalize(xvars[j][t_ind], self.minmax[j])

    def _eq_norm_11(m, t_ind, s_ind):
        return _normalize(xvars[j][t_ind, s_ind], self.minmax[j])
    
    _eq_norm = [_eq_norm_00, _eq_norm_01, _eq_norm_10, _eq_norm_11]

    xeq = []  # x expression; not named 'x' to avoid potential variable name conflicts
    for j in attributes:
        if ts[j] is None:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(rule=_eq_norm[switch[j]]))  # switch = 0
        elif len(ts[j]) == 1:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(ts[j][0], rule=_eq_norm[switch[j]]))  # switch = 1, 2
        else:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(ts[j][0], ts[j][1], rule=_eq_norm[switch[j]]))  # switch = 3
        xeq.append(m.component(f"GP{i}_x{j}"))

    # Define orders of polynomials, i.e., "basis" functions:

    n = []  # list of lists per attribute containing basis function orders used for that attribute
    for j in attributes:
        orders_j = np.unique(mtx[:, j])
        n.append(orders_j[orders_j != 0].tolist())
    
    def _orders(m, j):
        return n[j]
        
    m.add_component(f"GP{i}_orders", pyo.Set(attributes, initialize=_orders))  # orders of basis functions
    orders = m.component(f"GP{i}_orders")

    # Define polynomials, i.e., "basis" functions:

    def _basis(x, n):
        """Bernoulli polynomial, i.e., 'basis' function."""
        return self.phis[n][0] + sum(self.phis[n][k] * x ** k for k in range(1, len(self.phis[n])))
    
    def _eq_phi_00(m, order):
        return _basis(xeq[j], order - 1)

    def _eq_phi_01(m, s_ind, order):
        return _basis(xeq[j][s_ind], order - 1)
    
    def _eq_phi_10(m, t_ind, order):
        return _basis(xeq[j][t_ind], order - 1)
    
    def _eq_phi_11(m, t_ind, s_ind, order):
        return _basis(xeq[j][t_ind, s_ind], order - 1)

    _eq_phi = [_eq_phi_00, _eq_phi_01, _eq_phi_10, _eq_phi_11]

    phi = []
    for j in attributes:
        if ts[j] is None:
            m.add_component(f"GP{i}_phi{j}", pyo.Expression(orders[j], rule=_eq_phi[switch[j]]))  # switch = 0
        elif len(ts[j]) == 1:
            m.add_component(f"GP{i}_phi{j}", pyo.Expression(ts[j][0], orders[j], rule=_eq_phi[switch[j]]))  # switch = 1, 2
        else:
            m.add_component(f"GP{i}_phi{j}", pyo.Expression(ts[j][0], ts[j][1], orders[j], rule=_eq_phi[switch[j]]))  # switch = 3
        phi.append(m.component(f"GP{i}_phi{j}"))

    # Build GP expression:

    def _eq_y_X1(s_ind, t_ind=None):
        """GP expression, with 'scenarios'."""
        y = beta[s_ind, 0]  # initialize
        for term in range(1, len(terms)):  # == terms[1::]
            y_term = beta[s_ind, term]
            for j in attributes:
                order = mtx[term - 1, j]
                if order != 0:  # since 0 means none
                    if switch[j] == 0:
                        y_term *= phi[j][order]
                    elif switch[j] == 1:
                        y_term *= phi[j][s_ind, order]
                    elif switch[j] == 2:
                        y_term *= phi[j][t_ind, order]
                    elif switch[j] == 3:
                        y_term *= phi[j][t_ind, s_ind, order]
            y += y_term
        return y

    def _eq_y_X0(t_ind=None):
        """GP expression, without 'scenarios'."""
        y = beta[0]  # initialize
        for term in range(1, len(terms)):  # == terms[1::]
            y_term = beta[term]
            for j in attributes:
                order = mtx[term - 1, j]
                if order != 0:  # since 0 means none
                    if switch[j] == 0:
                        y_term *= phi[j][order]
                    elif switch[j] == 2:
                        y_term *= phi[j][t_ind, order]
            y += y_term
        return y

    def _eq_y_00(m):
        return _eq_y_X0()

    def _eq_y_01(m, s_ind):
        return _eq_y_X1(s_ind)
    
    def _eq_y_10(m, t_ind):
        return _eq_y_X0(t_ind)
    
    def _eq_y_11(m, t_ind, s_ind):
        return _eq_y_X1(s_ind, t_ind)

    _eq_y = [_eq_y_00, _eq_y_01, _eq_y_10, _eq_y_11]

    if ts[-1] is None:
        m.add_component(f"GP{i}_y", pyo.Expression(rule=_eq_y[switch[-1]]))  # switch = 0
    elif len(ts[-1]) == 1:
        m.add_component(f"GP{i}_y", pyo.Expression(ts[-1][0], rule=_eq_y[switch[-1]]))  # switch = 1, 2
    else:
        m.add_component(f"GP{i}_y", pyo.Expression(ts[-1][0], ts[-1][1], rule=_eq_y[switch[-1]]))  # switch = 3
    yeq = m.component(f"GP{i}_y")

    # Constraint of 'yvar' equal to 'm.GP#_y' Expression:
    
    def _constr_00(m):
        return yvar == yeq

    def _constr_01(m, s_ind):
        return yvar[s_ind] == yeq[s_ind]

    def _constr_10(m, t_ind):
        return yvar[t_ind] == yeq[t_ind]

    def _constr_11(m, t_ind, s_ind):
        return yvar[t_ind, s_ind] == yeq[t_ind, s_ind]

    _constr = [_constr_00, _constr_01, _constr_10, _constr_11]

    if ts[-1] is None:
        m.add_component(f"GP{i}", pyo.Constraint(rule=_constr[switch[-1]]))  # switch = 0
    elif len(ts[-1]) == 1:
        m.add_component(f"GP{i}", pyo.Constraint(ts[-1][0], rule=_constr[switch[-1]]))  # switch = 1, 2
    else:
        m.add_component(f"GP{i}", pyo.Constraint(ts[-1][0], ts[-1][1], rule=_constr[switch[-1]]))  # switch = 3

    return m

