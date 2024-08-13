"""

Acknowledgement:
    - https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/binaries.html#binary-packages
        - All technical papers, sales and publicity material resulting from use of the HSL codes within Ipopt must
        contain the following acknowledgement: HSL, a collection of Fortran codes for large-scale scientific
        computation. See http://www.hsl.rl.ac.uk.

"""
from FoKL import FoKLRoutines
import os
dir = os.path.abspath('')  # directory of notebook
# -----------------------------------------------------------------------
# UNCOMMENT IF USING LOCAL FOKL PACKAGE:
import sys
sys.path.append(os.path.join(dir, '..', '..'))  # package directory
from src.FoKL import FoKLRoutines
# -----------------------------------------------------------------------
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
import matplotlib.pyplot as plt
from pyomo.dae import Simulator


# =====================================================================
# =====================================================================
# LOAD AND PARSE SINE TEST DATA:

data = pd.read_csv(os.path.join(dir, "data", "tclab_sine_test.csv"))

tvec = data["Time"].values
Q1 = data["Q1"].values
TS1 = data["T1"].values

Q1f = interp1d(tvec, Q1, kind='previous')  # piecewise Q1
dQ1f_analytic = lambda t: 1500 * np.cos(30 * np.pi * t / tvec[-1]) * np.pi / tvec[-1]  # derivative of analytic Q1
dQ1f = interp1d(tvec, dQ1f_analytic(tvec), kind='previous')  # piecewise derivative of analytic Q1

# =====================================================================
# =====================================================================
# SMOOTHING OF SINE TEST:

window = 9  # odd number, mean at center +/- floor(window/2))

def smooth(TS1, window):
    """Apply centered average of size window."""
    TS1_smooth = np.zeros_like(TS1)
    w2 = int(np.floor(window / 2))
    w2p1 = w2 + 1

    # bleed in:
    for i in range(w2):
        TS1_smooth[i] = np.mean(TS1[:(i + w2p1)])

    # center:
    for i in range(w2, TS1_smooth.size - w2):
        TS1_smooth[i] = np.mean(TS1[(i - w2):(i + w2p1)])

    # bleed out:
    for i in range(-w2, 0):
        TS1_smooth[i] = np.mean(TS1[(i - w2)::])

    return TS1_smooth

TS1_smooth = smooth(TS1, window)

# =====================================================================
# =====================================================================
# DERIVATIVE OF SMOOTHED SINE TEST:

def gradient_h4(x, h):
    """h is step size. Order of error is h^4."""
    dx = np.zeros_like(x)

    # bleed in:
    h2 = 2 * h
    dx[0] = (x[1] - x[0]) / h
    dx[1] = (x[2] - x[0]) / h2

    # center difference:
    h12 = 12 * h
    for i in range(2, x.shape[0] - 2):
        dx[i] = (x[i - 2] - 8 * x[i - 1] + 8 * x[i + 1] - x[i + 2]) / h12
    
    # bleed out:
    dx[-2] = (x[-1] - x[-3]) / h2
    dx[-1] = (x[-1] - x[-2]) / h

    return dx

dTS1 = gradient_h4(TS1_smooth, tvec[1] - tvec[0])
dTS1f = interp1d(tvec, dTS1, kind='previous')  # piecewise, grab previous value

# =====================================================================
# =====================================================================
# GP MODEL OF DIFFERENTIATED SINE TEST:

try:
    GP_dT = FoKLRoutines.load(os.path.join("models", "pyomo_tclab_v7.fokl"))
except Exception as exception:
    GP_dT = FoKLRoutines.FoKL(kernel=1, UserWarnings=False, aic=True)
    GP_dT.fit([TS1_smooth, Q1, dQ1f(tvec)], dTS1, clean=True)
    GP_dT.save(os.path.join("models", "pyomo_tclab_v7.fokl"))

# =====================================================================
# =====================================================================
# CONTROLLER REFERENCE TRAJECTORY:

# time grid
t0 = 0
tf = 999
dt = 1
n = round(tf / dt)
t_grid = np.linspace(t0, tf, n + 1)

# ambient temperature
Tamb = 21.0

# setpoint/reference
def r(t):
    return np.interp(t, [0, 50, 150, 450, 550], [Tamb, Tamb, 60, 60, 35])

# =====================================================================
# =====================================================================
# OPTIMIZATION USING GP MODEL - CREATION OF PYOMO MODEL:

# Create a Pyomo model
m = pyo.ConcreteModel('TCLab Heater with GP Model')

# Define time domain
m.t = dae.ContinuousSet(bounds=(t0, tf))

# Define the state variables as a function of time
m.Ts1 = pyo.Var(m.t)

# Define the control variable (heater power) as a function of time
m.u1 = pyo.Var(m.t, bounds=(0, 100), initialize=50.0)  # != Q1f(t0)

# Define the derivative of the control variable
m.du1 = dae.DerivativeVar(m.u1, wrt=m.t)  # != dQ1f(t0)

# Define the derivatives of the state variables
m.dTs1 = dae.DerivativeVar(m.Ts1, wrt=m.t)

# Fix the initial conditions
m.Ts1[t0].fix(Tamb)

# Arguments to embed GP in Pyomo model:
xvars = [m.Ts1, m.u1, m.du1]
yvar = m.dTs1
draws = 5

# Embed GP:
GP_dT.to_pyomo(xvars, yvar, m, draws, with_blocks=False)

#============
#============
# Sim

if 1:  # Pyomo Simulator

    m.du1_dummy = pyo.Var(m.t)

    def _diffeq1(m, t):
        return m.du1[t] == m.du1_dummy[t]

    m.diffeq1 = pyo.Constraint(m.t, rule=_diffeq1)

    m.var_input = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    # m.var_input[m.u1] = {t0: 50}

    m.var_input[m.du1_dummy] = {t0: 0}
    m.var_input[m.y_avg] = {t0: 1}

    sim = Simulator(m, package='casadi')
    tsim, profiles = sim.simulate(
        numpoints=100, integrator='idas', varying_inputs=m.var_input
    )

    plt.plot(tsim, profiles)
    plt.show()

#============
#============
#============
#============

else:  # IPOPT

    # Define the integral of the squared error
    @m.Integral(m.t)
    def ise(m, t):
        return (r(t) - m.Ts1[t]) ** 2

    # Define the objective function
    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return m.ise

    # m.pprint()

    # =====================================================================
    # =====================================================================
    # OPTIMIZATION USING GP MODEL - SOLVER:

    # Apply a collocation method to numerically integrate the differential equations
    pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=100, wrt=m.t)

    # maybe loop over t steps in model; interp u based on csv sol from benchmark; fix u in this model to force 0 DoF --> simulate with IPOPT
    def u_ref(t):
        """estimate, not csv"""
        return np.interp(t,
                         [0, 30, 180, 200, 400],
                         [0, 100, 0, 60, 0])

    # for i, t in enumerate(m.t):
    #     m.u1[t].fix(u_ref(t))

    # Call our nonlinear optimization/equation solver, Ipopt
    solver = pyo.SolverFactory('ipopt')
    solver.options['linear_solver'] = 'ma27'
    solver.solve(m, tee=True)

    # Print solution

    tvec = m.t.data()

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot()
    axs[0, 0].plot(tvec, r(tvec))
    axs[0, 0].plot(tvec, m.Ts1[:]())
    axs[0, 0].legend(['r(t)', 'Ts1 (Var)'])

    axs[0, 1].plot(tvec, m.u1[:]())
    axs[0, 1].legend(['u1 (Var)'])

    axs[1, 1].plot(tvec, m.du1[:]())
    axs[1, 1].legend(['du1 (Var)'])

    axs[1, 0].plot(tvec, m.dTs1[:]())
    axs[1, 0].legend(['dTs1 (Var)'])

    plt.show()

b=1

