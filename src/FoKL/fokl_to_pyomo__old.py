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


# Internal functions:

def _process_arguments(self, xvars, yvar, m, draws, t_span, mtx, betas, minmax):
    """Process/format input arguments for 'to_pyomo' method."""
    if m is None:
        m = pyo.ConcreteModel('Global')

    if m.find_component('t') is None:
        if t_span is None:  # ODE not requested, so single index to avoid if-else statements in internal code
            m.t = pyo.Set(initialize=range(1))
        elif isinstance(t_span, list):
            m.t = dae.ContinuousSet(bounds=t_span)
        else:
            raise ValueError("Argument 't_span' must be a list of integration bounds.")
    elif t_span is not None:
        warnings.warn("Ignoring argument 't_span' because 'm.t' is already defined.", category=UserWarning)

    if not isinstance(xvars, list):  # if not list, make list
        xvars = [xvars]
    if isinstance(yvar, list):  # if list, make not list
        yvar = yvar[0]

    if minmax is None:
        minmax = self.minmax
        
    j = -1
    for xvar in xvars:
        j += 1
        if isinstance(xvar, str):  # else pre-defined Pyomo component, so ignore
            m.add_component(xvar, pyo.Var(m.t, domain=pyo.Reals, bounds=minmax[j]))
            xvars[j] = m.component(xvar)

    if isinstance(yvar, str):  # else pre-defined Pyomo component, so ignore
        m.add_component(yvar, pyo.Var(m.t, domain=pyo.Reals))
        yvar = m.component(yvar)

    if draws is None:
        draws = self.draws

    if mtx is None:
        mtx = self.mtx

    if betas is None:
        betas = self.betas

    return self, xvars, yvar, m, draws, t_span, mtx, betas, minmax


def _gp_as_pyomo(name, tvec, phis, draws, mtx, betas, xvars, minmax, model=None, scenarios=None):
    """tvec == m.t"""
    # Initialize sub-model for GP:
    if model is None:
        mGP = pyo.ConcreteModel(name)
        _with_block = True
    else:
        mGP = model
        _with_block = False

    # Some constants:
    mtx = np.array(mtx, dtype=int)  # indices/orders of basis functions (where 1 is B1 and 0 means none)

    # Some sets:
    mGP.terms = pyo.Set(initialize = range(mtx.shape[0] + 1))  # terms (including beta0)
    mGP.orders = pyo.Set(initialize = np.unique(mtx[mtx != 0]))  # orders of basis functions
    mGP.attributes = pyo.Set(initialize = range(mtx.shape[1]))  # input variables
    if scenarios is not None:
        if len(scenarios) == draws:
            raise NotImplementedError("Currently, length of 'scenarios' must equal 'draws'. In other words, 'scenarios' must index each of FoKL's 'draws'.")
        mGP.draws = scenarios  # using 'scenarios' instead of 'range(draws)' allows 'm.s' to be strings, etc.
    else:  # if 'scenarios' is None
        mGP.draws = pyo.Set(initialize = range(draws))

    if len(scenarios) == 1 or scenarios is None:  # then average draws into single Pyomo scenario


    elif len(scenarios) == draws:  # then each draw is scenario

    else:
        raise NotImplementedError()

    # COMMENTS:
    #   - if s None
    #       - m.s_temp = range(1)  # placeholder index
    #   - if len(s or m.s_temp) == 1  # (and draws != 1 ... else require draws > 1)
    #       - avg draws into single scenario
    #   - elif len(s) == draws
    #       - then each draw is scenario
    #   - else not implemented

    # ================
    # RTW:


    # Define beta coefficients:
    mGP.beta = pyo.Param(mGP.draws, mGP.terms, mutable=True)  # mutable=True, to change the value dynamically
    mGP.beta_avg = pyo.Param(mGP.terms, mutable=True)
    fix_betas(mGP, betas)

    # Define expression of normalized attributes (i.e., input variables):
    
    # TWO VERSIONS OF EQ_NORM / OR SWITCH CASE IN SINGLE FUNC:

    def _eq_norm(mGP, t, j):
        """Normalization constraint. (scenarios is None) ... modify slightly to index m.s_temp==1"""
        return (xvars[j][t] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])
    
    def _eq2_norm(mGP, t, j):
        """Normalization constraint. (scenarios is not None) ... index xvars by m.s"""
        return (xvars[j][t] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    # mGP.x = pyo.Expression(tvec, mGP.attributes, rule=_eq_norm)
    mGP.x = pyo.Expression(tvec, scenarios, mGP.attributes, rule=_eq_norm)

    # ===================================================================
    # Define polynomials (i.e., "basis" functions):
    
    nj = []  # list of [order, attribute] combinations used in GP
    for attribute in mGP.attributes:
        orders_j = np.unique(mtx[:, attribute])
        if any(orders_j != 0):
            for order_j in orders_j[orders_j != 0]:
                nj.append([order_j, attribute])
    
    # X NEEDS TO BE INDEXED BY m.s

    def _eq_phi(mGP, t, n, j):
        """FoKL's 'basis' functions."""
        nm1 = n - 1  # Python indexing, since n=1 refers to B1 which is phis[0]
        return phis[nm1][0] + sum(phis[nm1][k] * mGP.x[t, j] ** k for k in range(1, len(phis[nm1])))

    # INDEX BY m.s

    mGP.phi = pyo.Expression(tvec, nj, rule=_eq_phi)

    # ===================================================================
    # Build GP expression:

    # Draws:

    # CONSIDER AVERAGING ALL TO PLACE IN SINGLE SCENARIO, OR EACH DRAW IN EACH SCENARIO (IN WHICH CASE NO Y_AVG NEEDED)

    def _eq_y(mGP, t, draw):
        """FoKL's GP equation."""
        y = mGP.beta[draw, 0]  # initialize
        
        for term in range(1, len(mGP.terms)):  # == m.terms[1::]
            y_term = mGP.beta[draw, term]

            for j in mGP.attributes:
                n = mtx[term - 1, j]

                if n != 0:  # since 0 means none
                    y_term *= mGP.phi[t, n, j]

            y += y_term

        return y

    mGP.y = pyo.Expression(tvec, mGP.draws, rule=_eq_y)

    # Average (IGNORE FOR NOW BECAUSE m.s):

    # def _eq_y_avg(mGP, t):
    #     """FoKL's GP equation, averaged across draws."""
    #     y = mGP.beta_avg[0]  # initialize
        
    #     for term in range(1, len(mGP.terms)):  # == m.terms[1::]
    #         y_term = mGP.beta_avg[term]

    #         for j in mGP.attributes:
    #             n = mtx[term - 1, j]

    #             if n != 0:  # since 0 means none
    #                 y_term *= mGP.phi[t, n, j]

    #         y += y_term

    #     return y

    # mGP.y_avg = pyo.Expression(tvec, rule=_eq_y_avg)

    # Standard deviation (IGNORE FOR NOW):

    # def _eq_y_std(mGP, t):
    #     """Standard deviation of draws from FoKL's GP equation."""
    #     return sqrt(sum(mGP.y[t, draw] ** 2 for draw in mGP.draws) / len(mGP.draws) + 1e-9)

    # mGP.y_std = pyo.Expression(tvec, rule=_eq_y_std)

    return mGP


# End internal functions.
# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
# Module functions:

def fix_betas(mGP, betas):
    """
    Fix the already-initialized Pyomo beta parameters to scalar values in 'betas', using last 'betas' draw as first Pyomo draw. Include average.
    
    | Argument | Type        | Description                                                                       |
    |----------|-------------|-----------------------------------------------------------------------------------|
    | mGP      | Pyomo model | sub-model containing the GP, i.e., 'm.GP#' where # is the integer index of the GP |
    | betas    | ndarray     | [draws x terms] beta coefficients of GP model                                     |
    
    | Output | Type        | Description                                                                                     |
    |--------|-------------|-------------------------------------------------------------------------------------------------|
    | mGP    | Pyomo model | sub-model 'm.GP#' with 'm.GP#.beta' and 'm.GP#.beta_avg' parameters now with their values fixed |

    """
    # Update value of 'beta_avg' Param:
    betas_avg = np.mean(betas[-len(mGP.draws)::, :], axis=0)
    for term in mGP.terms:
        mGP.beta_avg[term] = betas_avg[term]

        # Update value of 'beta' Param:
        for draw in mGP.draws:
            mGP.beta[draw, term] = betas[-(draw + 1), term]


def fokl_to_pyomo(self, xvars, yvar, m=None, draws=None, t_span=None, mtx=None, betas=None, minmax=None, with_blocks=False, scenarios=None):
    """
    Convert GP model from FoKL class to Pyomo model.
    
    | Argument    | Type                                       | Description                                                                                                                                                                              |
    |-------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | self        | FoKL class                                 | FoKL class object with Bernoulli Polynomials kernel, 'FoKLRoutines.FoKL(kernel=1)'                                                                                                       |
    | xvars       | list of str, Pyomo components, and/or None | list of GP input variables; if str then defaults to 'pyo.Var()', else pre-define 'dae.DerivativeVar()', etc. in 'm' and pass component directly in list; 'None' to later define manually |
    | yvar        | str, Pyomo component, and/or None          | GP output variable; behaves like 'xvars'                                                                                                                                                 |
    | m           | Pyomo model                                | pre-defined Pyomo model                                                                                                                                                                  |
    | draws       | int                                        | number of GP draws to embed in Pyomo                                                                                                                                                     |
    | t_span      | list of two floats                         | integration time [t0, tf]; used to define 'm.t = dae.ContinuousSet(bounds=t_span)' if 'm.t' not defined, else 't_span' is ignored                                                        |
    | mtx         | ndarray                                    | GP's interaction matrix                                                                                                                                                                  |
    | betas       | ndarray                                    | GP's coefficients                                                                                                                                                                        |
    | minmax      | list of lists of two floats                | GP's normalization of input variables [[min, max], ..., [min, max]]                                                                                                                      |
    | with_blocks | boolean                                    | to define the GP's Pyomo components as a Pyomo block (i.e., sub-model) of the main Pyomo model 'm'                                                                                       |
    | scenarios   | Pyomo set                                  | Pyomo scenarios over which to optimize; i.e., FoKL draws to optimize individually                                                                                                        |
    
    | Output | Type        | Description                                        |
    |--------|-------------|----------------------------------------------------|
    | m      | Pyomo model | input argument 'm' with 'self' embedded as 'm.GP#' |
    
    Reserved Pyomo components:
        - m.t
        - m.GP#, where # is integer index of GP beginning at 0
        - m.[xvar] for xvar in xvars
        - m.[yvar]
        
    Tips:
        - 'xvars', if user-defined, should be bounded by 'minmax' of FoKL model 'GP';
          this is not technically required since the GP is a polynomial, but is recommended because the GP is not intended to extrapolate.
            - 'm.[xvars[j]].setlb(GP.minmax[j][0])'
            - 'm.[xvars[j]].setub(GP.minmax[j][1])'

    TO-DO / FUTURE DEV.:
        - 'with_blocks=False' SHOULD YIELD 'm.GP#_beta', etc.; CURRENTLY, 'm.beta' MEANS MULTIPLE GPs CANNOT BE SUPPORTED WITHOUT BLOCKS
    
    """
    if with_blocks is True and m is None:
        raise NotImplementedError()

    # Process input arguments:
    self, xvars, yvar, m, draws, t_span, mtx, betas, minmax = _process_arguments(self, xvars, yvar, m, draws, t_span, mtx, betas, minmax)

    # Find next available GP index:
    if with_blocks is True:
        i = 0
        while m.find_component(f"GP{i}") is not None:
            i += 1

    # Create Pyomo model with GP:
    if with_blocks is False:
        model = m
        gp_name = 'GP Model'
    else:
        model = None
        gp_name = f"GP{i}"
    mGP = _gp_as_pyomo(gp_name, m.t, self.phis, draws, mtx, betas, xvars, minmax, model=model, scenarios=scenarios)

    # Set 'yvar' equal to GP:
    
    # IF s IS NONE --> yvar not indexed by m.s; else is; etc.  ---> maybe two versions of constr.

    def _constr_yvar(mGP, t):
        """Set 'yvar' equal to GP."""
        return yvar[t] == mGP.y_avg[t]
    
    mGP.constr_yvar = pyo.Constraint(m.t, rule=_constr_yvar)

    # Merge 'mGP' with global Pyomo model:
    if with_blocks is True:
        m.add_component(f"GP{i}", mGP)

        return m

    else:

        return mGP
    


    # -----------------
    #TODO
    #   - reproduce with s is None
    #   - then try s (with all xvars as [m.t, m.s])




