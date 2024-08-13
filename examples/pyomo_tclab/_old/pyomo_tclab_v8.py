"""

Change Log:
    - v8 like v7.py but including step test in training data

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
from scipy.interpolate import interp1d
from pyomo.dae import Simulator


# =====================================================================
# =====================================================================
# LOAD AND PARSE DATA:

data = [pd.read_csv(os.path.join(dir, "data", "tclab_sine_test.csv")),  # sine
        pd.read_csv(os.path.join(dir, "data", "tclab_step_test.csv"))]  # step

n = len(data)

tvec = list(df["Time"].values for df in data)
Q1 = list(df["Q1"].values for df in data)
TS1 = list(df["T1"].values for df in data)

Q1f = list(interp1d(tvec[i], Q1[i], kind='previous') for i in range(n))  # piecewise Q1
dQ1f_analytic = [lambda t: 1500 * np.cos(30 * np.pi * t / tvec[0][-1]) * np.pi / tvec[0][-1], 
                 lambda t: 0 * t]  # derivative of analytic Q1
dQ1f = list(interp1d(tvec[i], dQ1f_analytic[i](tvec[i]), kind='previous') for i in range(n))  # piecewise derivative of analytic Q1
dQ1f_values = list(dQ1f[i](tvec[i]) for i in range(n))

# =====================================================================
# =====================================================================
# SMOOTHING:

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

TS1_smooth = list(smooth(TS1_i, window) for TS1_i in TS1)

# =====================================================================
# =====================================================================
# DERIVATIVE:

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

dTS1 = []
dTS1f = []
for i in range(n):
    dTS1.append(gradient_h4(TS1_smooth[i], tvec[i][1] - tvec[i][0]))
    dTS1f.append(interp1d(tvec[i], dTS1[i], kind='previous'))  # piecewise, grab previous value

# =====================================================================
# =====================================================================
# GP MODEL OF DIFFERENTIATED SINE TEST:

filename = os.path.join("models", "pyomo_tclab_v8.fokl")
try:
    GP_dT = FoKLRoutines.load(filename)
except Exception as exception:
    GP_dT = FoKLRoutines.FoKL(kernel=1, UserWarnings=False, aic=True)
    GP_dT.fit([np.concatenate(TS1_smooth), np.concatenate(Q1), np.concatenate(dQ1f_values)],
              np.concatenate(dTS1), 
              clean=True)
    GP_dT.save(filename)

GP_dT.coverage3(plot='sorted')

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

# time points of setpoint/reference values
tr = [t0, 50, 150, 450, 550, tf]

# setpoint/reference
def r(t):
    return np.interp(t, tr, [Tamb, Tamb, 60, 60, 35, 35])

# derivative of setpoint/reference
dr_interp = interp1d(tr, [0, (60 - Tamb) / 100, 0, (35 - 60) / 100, 0, 0], kind='previous')
def dr(t):
    return float(dr_interp(t))

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
m.u1 = pyo.Var(m.t, bounds=(0, 100), initialize=50)  # != Q1f(t0)

# Define the derivative of the control variable
m.du1 = dae.DerivativeVar(m.u1, wrt=m.t)  # != dQ1f(t0)

# Define the derivatives of the state variables
m.dTs1 = dae.DerivativeVar(m.Ts1, wrt=m.t)

# Fix the initial conditions
m.u1[t0].fix(0)
m.Ts1[t0].fix(Tamb)

# Arguments to embed GP in Pyomo model:
xvars = [m.Ts1, m.u1, m.du1]
yvar = m.dTs1
draws = 5

# Embed GP:
GP_dT.to_pyomo(xvars, yvar, m, draws, with_blocks=False)

# =====================================================================
# =====================================================================
# OPTIMIZATION USING GP MODEL - SOLVER:

# Fix u1 to benchmark solution, forcing 0 DoF for sake of debugging:

u1_benchmark = np.loadtxt(os.path.join(dir, 'data', 'u1_benchmark_solution.csv'), delimiter=',')
u1_benchmark = np.concatenate([np.array([t0, 0])[np.newaxis, :],
                               u1_benchmark[1:107, :],
                               u1_benchmark[130:218, :],
                               u1_benchmark[243:354, :],
                               u1_benchmark[370:-1, :]], axis=0)  # remove oscillations

def u1_ref(t):
    """u1 values from benchmark solution."""
    u1 = np.interp(t, u1_benchmark[:, 0], u1_benchmark[:, 1])
    if u1 < 0:
        return 0
    elif u1 > 100:
        return 100
    else:
        return u1

du1_benchmark = np.array([u1_benchmark[:, 0], np.gradient(u1_benchmark[:, 1], u1_benchmark[:, 0])]).T

def du1_ref(t):
    """du1 values via u1 from benchmark solution."""
    return np.interp(t, du1_benchmark[:, 0], du1_benchmark[:, 1])

# =====================================================================
# SOLVE WITH SIMULATOR:

if 0:

    # To enable dae.Simulator to run, the DerivativeVar's need to exist as the LHS of a Constraint.
    # m.dTs1 already satisifies this, with the m.y_avg Expression from FoKL as the RHS.
    # Creating a dummy so that m.du1 may also satisfy this:
    m.du1_dummy = pyo.Var(m.t)

    @m.Constraint(m.t)
    def constr_dudt(m, t):
        return m.du1[t] == m.du1_dummy[t]

    # Then, the RHS's need to be initialized:
    m.var_input = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.var_input[m.du1_dummy] = {}  # == m.du1, defined via 'u1_benchmark'
    m.var_input[m.y_avg] = {}  # == m.dTs1, defined to match the control reference trajectory
    for t in du1_benchmark[:, 0]:
        m.var_input[m.du1_dummy].update({t: du1_ref(t)})
    for t in tr:
        m.var_input[m.y_avg].update({t: dr(t)})

    # Simulate:
    sim = Simulator(m, package='casadi')
    tsim, profiles = sim.simulate(
        numpoints=100, integrator='idas', varying_inputs=m.var_input
    )

    # Plot:
    plt.plot(tsim, profiles)
    plt.title('Simulation Results, dTs1 = d(r)/dt, du1[t0] = 0')
    plt.xlabel('Time (s)')
    plt.legend(['Ts1 (°C)', 'u1 (%)'])
    plt.grid()
    plt.show()

# =====================================================================
# SOLVE WITH IPOPT:

else:
    from pyomo.environ import *

    # Define the error model
    @m.Integral(m.t)
    def ise(m, t):
        # return (r(t) - m.Ts1[t]) ** 2  # == ise (integral of squared error); great ramp up but failed ramp down
        # return abs(r(t) - m.Ts1[t])  # fail
        return exp(abs(r(t) - m.Ts1[t]))  # jumpy but gets ramp down; may be worth investigation
        # return 1.001 ** abs(r(t) - m.Ts1[t])  # fail
        # return sqrt((r(t) - m.Ts1[t]) ** 2)  # == sqrt(ise) = abs(error); fail
        # return (dr(t) - m.dTs1[t]) ** 2  # == ise, with GP; fail
        # return (u1_ref(t) - m.u1[t]) ** 2  # fail

    # Define the objective function
    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return m.ise

    # Apply a collocation method to numerically integrate the differential equations
    pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=100, wrt=m.t)

    # Provide initial guess from benchmark solution:
    # for t in m.t:
    #     # m.y_avg[t] = dr(t)  # forces Ts1 to match r(t), but u1 is unaffected
    #     # m.dTs1[t].fix(dr(t))  # Ts1 gets shape but has large "stepwise" jumps in value
    #     # m.u1[t].fix(u1_ref(t))  # Ts1 not accurate
    #     # m.u1[t] = u1_ref(t)  # initialized but not fixed; same solution as without this initialization
    #     m.du1[t] = du1_ref(t)  #

    # Call our nonlinear optimization/equation solver, Ipopt
    solver = pyo.SolverFactory('ipopt')
    solver.options['linear_solver'] = 'ma27'  # ma57
    solver.solve(m, tee=True)

    # Plot solution

    tvec = m.t.data()

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot()
    axs[0, 0].plot(tvec, r(tvec))
    axs[0, 0].plot(tvec, m.Ts1[:]())
    axs[0, 0].legend(['r(t)', 'Ts1 (Var)'])

    axs[0, 1].plot(u1_benchmark[:, 0], u1_benchmark[:, 1])
    axs[0, 1].plot(tvec, m.u1[:]())
    axs[0, 1].legend(['u1 (benchmark)', 'u1 (Var)'])

    axs[1, 1].plot(du1_benchmark[:, 0], du1_benchmark[:, 1])
    axs[1, 1].plot(tvec, m.du1[:]())
    axs[1, 1].legend(['du1 (benchmark)', 'du1 (Var)'])

    axs[1, 0].plot(tvec, m.dTs1[:]())
    axs[1, 0].plot(tvec, m.y_avg[:]())
    axs[1, 0].plot(tvec, GP_dT.evaluate([m.Ts1[:](), m.u1[:](), m.du1[:]()], clean=True))  # GP confirmation
    axs[1, 0].legend(['dTs1 (Var)', 'm.y_avg', 'dTs1 (GP check)'])

    for i in range(2):
        for j in range(2):
            axs[i, j].grid()

    # plt.savefig(os.path.join(dir, 'data', 'pyomo_tclab_v8_ipopt.png'))
    plt.show()

