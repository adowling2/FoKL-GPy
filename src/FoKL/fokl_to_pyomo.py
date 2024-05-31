import pyomo.environ as pyo
import numpy as np
import warnings
import copy
from pyomo.environ import *


def _check_models(models):
    """Check 'models' is list of class object(s); return False if cannot be resolved."""
    if not isinstance(models, list):
        if isinstance(models, object):  # single model; make list
            models = [models]
        else:
            return False

    if all(isinstance(model, object) for model in models):
        return models
    else:
        return False


def _check_xvars(xvars):
    """Check 'xvars' is list of list(s) of string(s); return False if cannot be resolved."""
    if isinstance(xvars, str):
        xvars = [[xvars]]
    if isinstance(xvars[0], str):
        xvars = [xvars]

    if isinstance(xvars, list) and isinstance(xvars[0], list) \
            and all(isinstance(xvar[j], str) for xvar in xvars for j in range(len(xvar))):
        return xvars
    else:
        return False


def _check_yvars(yvars):
    """Check 'yvars' is list of string(s); return False if cannot be resolved."""
    if isinstance(yvars, str):
        yvars = [yvars]

    if isinstance(yvars, list) and all(isinstance(yvars[i], str) for i in range(len(yvars))):
        return yvars
    else:
        return False
    

def _handle_exceptions(models, xvars, yvars, draws, m, xfix, yfix, truescale):
    """Check all inputs to automatically format, throw errors or warnings, etc."""
    xvars_true = None  # define variable name since returned even if not defined
    i_norm = None

    # Check inputs:

    models = _check_models(models)
    if models is False:
        raise ValueError("'models' must be a list of FoKL model class object(s).")

    xvars = _check_xvars(xvars)
    if xvars is False:
        raise ValueError("'xvars' must be a list of list(s) of string(s).")

    yvars = _check_yvars(yvars)
    if yvars is False:
        raise ValueError("'yvars' must be a list of string(s).")

    # Further check inputs:

    n = len(models)

    def _error_align(input_varname):
        raise ValueError(f"'models' and '{input_varname}' must align.")

    if len(xvars) != n or any(len(xvars[i]) != models[i].inputs.shape[1] for i in range(n)):
        _error_align('xvars')

    if len(yvars) != n:
        _error_align('yvars')

    if draws is None:
        draws = []
        for model in models:
            draws.append(model.draws)
    elif isinstance(draws, int):  # then use single 'draws' value for all models
        draws = [draws] * n
    elif isinstance(draws, list):  # then confirm same length as models
        if len(draws) != n:
            _error_align('draws')

    if m is None:
        m = pyo.ConcreteModel()
    else:
        # check for overlapping variable names
        # ...
        # get other pyomo model info
        pass

    if xfix is None:
        xfix = [None] * n
    # else:  # assume properly formatted, e.g., 'xfix=[[0.2, None, 0.6], ..., [None], ..., [0.5, 0.1]]'

    if yfix is None:
        yfix = [None] * n
    # else:  # assume properly formatted, e.g., 'yfix=[342, ..., None, ..., 107]'

    # Check 'truescale':
    if isinstance(truescale, bool):  # make all inputs for all models true
        truefalse = copy.copy(truescale)
        truescale = []
        for im in range(n):
            truescale.append([truefalse] * models[im].inputs.shape[1])
    elif isinstance(truescale, list):
        for im in range(n):
            if isinstance(truescale[im], bool):  # make all inputs for current model true
                truescale[im] = [truescale[im]] * models[im].inputs.shape[1]
    
    # # Adjust 'xvars' based on 'truescale' to define two Pyomo variables (one normalized, one truescale):
    # if any(truescale[im] for im in range(n)) is True:  # if any input in current model is truescale
    #     xvars_true = copy.deepcopy(xvars)  # copy 'xvars'; then make old 'xvars' where 'truescale=True' into, e.g., "P_nomalized" from "P"
    #     i_norm = []  # indices of true scale input variables in all models
    #     for im in range(n):
    #         i_norm_im = []  # indices of true scale input variables in current model
    #         for j in range(models[im].inputs.shape[1]):  # for input var in input vars
    #             if truescale[im][j] is True:
    #                 i_norm_im.append(j)
    #                 xvars[im][j] = f"{xvars[im][j]}_normalized"  # create new variable name for use in FoKL equation; the original variable name will be true scale
    #         i_norm.append(i_norm_im)

    for model in models:
        try:
            if model.kernel != 'Bernoulli Polynomials':
                warnings.warn("'kernel' should be 'Bernoulli Polynomials'. The kernel is being switched for Pyomo but "
                              "it is highly recommended to retrain the model.", category=UserWarning)
        except Exception as exception:
            pass  # assume user did not train model but is manually passing 'betas', 'mtx', 'draws' in model(s)

    # return models, xvars, yvars, draws, m, xfix, yfix, truescale, xvars_true, i_norm
    return models, xvars, yvars, draws, m, xfix, yfix, truescale


def _add_gp(self, xvars, yvar, draws, m, xfix, yfix, truescale, igp):
    """
    Add GP to Pyomo model.
    
    - assuming all inputs are properly formatted since passed here internally.
    - assuming all 'm.GP#_xxxx' components are available
    """
    # Define some constants:
    
    t = np.array(self.mtx - 1, dtype=int)  # indices of polynomial (where 0 is B1 and -1 means none)
    lt = t.shape[0] + 1  # length of terms (including beta0)
    lv = t.shape[1]  # length of input variables

    # Define some Pyomo sets (and indexed var):

    m.add_component(f"GP{igp}_scenarios", pyo.Set(initialize=range(draws)))  # index for scenario (i.e., FoKL draw)
    m.add_component(f"GP{igp}_j", pyo.Set(initialize=range(lv)))  # index for FoKL input variable
    m.add_component(f"GP{igp}_k", pyo.Set(initialize=range(lt)))  # index for FoKL term (where 0 is beta0)
    m.add_component(f"GP{igp}_b", pyo.Var(m.component(f"GP{igp}_scenarios"), m.component(f"GP{igp}_k")))  # FoKL coefficients (i.e., betas)

    # Define FoKL output (and its counterparts) as Pyomo variable:

    if m.find_component(yvar) is None:  # then define; else a previous model already defined this variable
        m.add_component(yvar, pyo.Var(within=pyo.Reals))  # FoKL output, as single variable equal across GP's

    m.add_component(f"GP{igp}_{yvar}_draw", pyo.Var(m.component(f"GP{igp}_scenarios"), within=pyo.Reals))  # FoKL output, evaluated at draw for current GP
    m.add_component(f"GP{igp}_{yvar}_mean", pyo.Var(within=pyo.Reals))  # FoKL output, mean of draws for current GP
    m.add_component(f"GP{igp}_{yvar}_std", pyo.Var(within=pyo.Reals))  # FoKL output, standard deviation of draws for current GP

    # Define FoKL normalized inputs (and their true scale counterparts) as Pyomo variables:

    for j in m.component(f"GP{igp}_j"):
        m.add_component(f"GP{igp}_{xvars[j]}_norm", pyo.Var(within=pyo.Reals, bounds=[0, 1], initialize=0.0))  # FoKL input variables

        if truescale[j] is True:  # create expression relating normalized variable to true scale
            if m.find_component(xvars[j]) is None:  # confirm truescale variable was not previously defined in prior GP
                m.add_component(xvars[j], pyo.Var(within=pyo.Reals, bounds=self.minmax[j], initialize=self.minmax[j][0]))
            else:  # already exists, but update bounds if this GP's minmax are more limiting
                m.component(xvars[j])._domain = pyo.Reals  # within

                if self.minmax[j][0] > m.component(xvars[j]).bounds[0]:  # if current lower bound is higher, then use as limiting case
                    m.component(xvars[j]).setlb(self.minmax[j][0])
                    m.component(xvars[j]).set_value(self.minmax[j][0])  # initialize
                
                if self.minmax[j][1] < m.component(xvars[j]).bounds[1]:  # if current upper bound is lower, then use as limiting case
                    m.component(xvars[j]).setub(self.minmax[j][1])

        else:  # set normalized variable EQUAL to true scale, since user specified truescale[j]=False
            if m.find_component(xvars[j]) is None:
                m.add_component(xvars[j], pyo.Var())  # to be set equal to '{xvar}_norm' in later constraint (where normalization constraint is applied)

    # Define basis functions:

    ni_ids = []  # orders of basis functions used (where 0 is B1), per term
    basis_nj = []  # for future use when indexing 'm.GP#_basis'
    for j in range(lv):  # for input variable in input variables
        ni_ids.append(np.sort(np.unique(t[:, j][t[:, j] != -1])).tolist())
        for n in ni_ids[j]:  # for order of basis function in unique orders, per current input variable
            basis_nj.append([n, j])

    def symbolic_basis(m):
        """Basis functions as symbolic. See 'evaluate_basis' for source of equation."""
        for [n, j] in basis_nj:
            m.component(f"GP{igp}_basis")[n, j] = self.phis[n][0] + sum(self.phis[n][k] * (m.component(f"GP{igp}_{xvars[j]}_norm") ** k)
                                                    for k in range(1, len(self.phis[n])))
        return

    m.add_component(f"GP{igp}_basis", pyo.Expression(basis_nj))  # create indices for required basis functions
    symbolic_basis(m)  # may be better to write as rule, but 'pyo.Expression(basis_nj, rule=symbolic_basis)' failed

    # Define FoKL equation (for each draw, where betas are different):

    for i in m.component(f"GP{igp}_scenarios"):  # for scenario (i.e., draw) in scenarios (i.e., draws)
        for k in m.component(f"GP{igp}_k"):  # for term in terms
            m.component(f"GP{igp}_b")[i, k].fix(self.betas[-(i + 1), k])  # define values of betas, with i=0 as last FoKL draw

    def symbolic_fokl(m):
        """FoKL models (i.e., scenarios) as symbolic, assuming 'Bernoulli Polynomials."""
        for i in m.component(f"GP{igp}_scenarios"):  # for scenario (i.e., draw) in scenarios (i.e., draws)
            m.component(f"GP{igp}_expr")[i] = m.component(f"GP{igp}_b")[i, 0]  # initialize with beta0
            for k in range(1, lt):  # for term in non-zeros terms (i.e., exclude beta0)
                tk = t[k - 1, :]  # interaction matrix of current term
                tk_mask = tk != -1  # ignore if -1 (recall -1 basis function means none)
                if any(tk_mask):  # should always be true because FoKL 'fit' removes rows from 'mtx' without basis
                    term_k = m.component(f"GP{igp}_b")[i, k]
                    for j in m.component(f"GP{igp}_j"):  # for input variable in input variables
                        if tk_mask[j]:  # for variable in term
                            term_k *= m.component(f"GP{igp}_basis")[tk[j], j]  # multiply basis function(s) with beta to form term
                else:
                    term_k = 0
                m.component(f"GP{igp}_expr")[i] += term_k  # add term to expression
        return

    m.add_component(f"GP{igp}_expr", pyo.Expression(m.component(f"GP{igp}_scenarios")))  # FoKL models (i.e., scenarios, draws)
    symbolic_fokl(m)  # may be better to write as rule

    # Apply constraint equating 'yvar' draws with FoKL equation draws:

    def symbolic_scenario(m):
        """Define each scenario, meaning a different draw of 'betas' for y=f(x), as a constraint."""
        for i in m.component(f"GP{igp}_scenarios"):
            m.component(f"GP{igp}_draw_constr")[i] = m.component(f"GP{igp}_{yvar}_draw")[i] == m.component(f"GP{igp}_expr")[i]
        return

    m.add_component(f"GP{igp}_draw_constr", pyo.Constraint(m.component(f"GP{igp}_scenarios")))  # set of constraints, one per scenario
    symbolic_scenario(m)  # may be better to write as rule

    # Define mean and standard deviation equations:

    m.add_component(f"GP{igp}_mean_constr", pyo.Constraint(expr=
        m.component(f"GP{igp}_{yvar}_mean") == sum(m.component(f"GP{igp}_{yvar}_draw")[i] for i in m.component(f"GP{igp}_scenarios")) / draws
    ))

    # --------------------------------------------------------------------------------------
    # [IN DEV; sqrt(0) cannot return derivative]:

    # m.add_component(f"GP{igp}_std_constr", pyo.Constraint(expr=
    #     m.component(f"GP{igp}_{yvar}_std") == sqrt(sum((m.component(f"GP{igp}_{yvar}_draw")[i] - m.component(f"GP{igp}_{yvar}_mean")) ** 2 for i in m.component(f"GP{igp}_scenarios")) / (draws - 1))
    # ))

    # [END DEV].
    # ---------------------------------------------------------------------------------------

    # Append constraint of 'yvar' equaling mean:

    if m.component(f"{yvar}_constr") is None:  # define if not yet defined by previous GP
        m.add_component(f"{yvar}_constr", pyo.ConstraintList())

    m.component(f"{yvar}_constr").add(
        m.component(yvar) == m.component(f"GP{igp}_{yvar}_mean")
    )

    # Add normalization constraint (or set equal) for each input variable:

    def symbolic_normalize(m):
        """Relate normalized and true scale input variable."""
        for j in m.component(f"GP{igp}_j"):
            if truescale[j] is True:
                m.component(f"GP{igp}_norm_constr")[j] = m.component(xvars[j]) == m.component(f"GP{igp}_{xvars[j]}_norm") * (self.minmax[j][1] - self.minmax[j][0]) + self.minmax[j][0]  # normalization scale
            else:
                m.component(f"GP{igp}_norm_constr")[j] = m.component(xvars[j]) == m.component(f"GP{igp}_{xvars[j]}_norm")  # set equal
        return

    m.add_component(f"GP{igp}_norm_constr", pyo.Constraint(m.component(f"GP{igp}_j")))
    symbolic_normalize(m)  # may be better to write as rule

    # If user has specified certain values for inputs or output within the FoKL equation, fix normalized input variables and/or fix draws:

    if xfix is not None:
        for j in m.component(f"GP{igp}_j"):
            if xfix[j] is not None:
                # m.component(xvars[j]).fix(xfix[j])  # commented out because assuming user only wants variables WITHIN the FoKL equation to be fixed
                
                if truescale[j] is True:  # normalize value before fixing
                    xfix[j] = (xfix[j] - self.minmax[j][0]) / (self.minmax[j][1] - self.minmax[j][0])
                
                m.component(f"GP{igp}_{xvars[j]}_norm").fix(xfix[j])  # fix normalized variable to normalized value
    
    if yfix is not None:
        for i in m.component(f"GP{igp}_scenarios"):
            m.component(f"GP{igp}_{yvar}_draw")[i].fix(yfix)  # fix draws of output variable

    # Return Pyomo model with current FoKL model appended:

    return m


def fokl_to_pyomo(models, xvars, yvars, draws=None, m=None, xfix=None, yfix=None, truescale=True):
    """
    
    'to_pyomo' passes inputs to here;
    user may use this for multiple GP's at once (so symbolic bases get defined once, and so 'xvars' can be repeated);

    - 'truescale' changes where 'xfix' gets defined such that 'xfix' for 'truescale=True' must be  entered as true scale;
    - if repeating any 'xvars' across models, the 'truescale' value for the first time it is defined will be used so be
      careful to ensure repeat 'xvars' do not have differently intended 'truescale' values;

    """
    # Process inputs:
    # models, xvars, yvars, draws, m, xfix, yfix, truescale, xvars_true, i_norm = _handle_exceptions(models, xvars, yvars, draws, m, xfix, yfix, truescale)
    models, xvars, yvars, draws, m, xfix, yfix, truescale = _handle_exceptions(models, xvars, yvars, draws, m, xfix, yfix, truescale)

    # Initialize Pyomo model:
    if m is None:
        m = pyo.ConcreteModel()

    # Find index for next available GP (in case already defined), then pass to function that defines GP in Pyomo:
    igp = 0  # initial index of GP; increases until next available GP
    for im in range(len(models)):  # index of model
        while m.find_component(f"GP{igp}_expr") is not None:  # GP at this index already exists
            igp += 1  # increase index
        
        # Add GP to Pyomo:
        # m = _add_gp(models[im], xvars[im], yvars[im], draws[im], m, xfix[im], yfix[im], truescale[im], igp, xvars_true[im])
        m = _add_gp(models[im], xvars[im], yvars[im], draws[im], m, xfix[im], yfix[im], truescale[im], igp)

    # Return Pyomo model with all FoKL models embedded:

    return m

