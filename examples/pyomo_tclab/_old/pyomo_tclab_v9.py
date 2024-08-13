from FoKL import FoKLRoutines
import os
dir = os.path.abspath('')  # directory of notebook
# -----------------------------------------------------------------------
# UNCOMMENT IF USING LOCAL FOKL PACKAGE:
import sys
sys.path.append(os.path.join(dir, '..', '..'))  # package directory
from src.FoKL import FoKLRoutines
# -----------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pyomo.dae as dae


# System-level parameters:

Tamb = 21  # ambient temperature (C)
Tmax = 110 - Tamb  # maximumum temperature, i.e., steady state of u=100; 100-Tamb so that u=50 yields 55 C like data
t0 = 0  # start time (s)
dt = 1  # time step (s)
t_ramp = 1000  # estimated time to reach steady state (s)
t_rest = 50  # time to "rest" at steady state for sake of obtaining dT=0 training data
t_drop = 800  # estimated time to return to Tamb (s)
n = 20  # number of step tests to obtain data for

# =====================================================================
# =====================================================================
# Generate data for demonstration:

u1_tests = np.linspace(100 / n, 100, n)  # heater power (%), const. values at which to obtain data
T_tests = np.linspace(Tamb + (Tmax - Tamb) / n, Tmax, n)  # steady state temperature corresponding to u1_tests

t_seg = t_ramp + t_rest + t_drop + t_rest / n  # length of single test
tf = t0 + n * t_seg
tvec = np.linspace(t0, tf, int((tf + 1) / dt))
dt = tvec[1] - tvec[0]  # redefine in case dt got rounded

u = np.zeros_like(tvec)
T = np.zeros_like(tvec) + Tamb

def T_approx(t_start, t_end, i):
    """Estimate temperature change using natural log of quadratic."""
    x = tvec[t_start:t_end] - tvec[t_start]
    H = T_tests[i] - Tamb
    W = t_end - t_start
    
    W0 = W / np.sqrt(1 - 1 / H)                 # == W'
    Q = H * (1 - ((x - W + W0) / W0 - 1) ** 2)  # == Q'
    L = np.log(Q) * H / np.log(H)               # == L'
    
    return L

t1 = int(t0)
for i in range(n):
    t2 = int(t1 + t_ramp)
    t3 = int(t1 + t_ramp + t_rest)
    t4 = int(t1 + t_ramp + t_rest + t_drop)
    
    T[t1:t2] = Tamb + T_approx(t1, t2, i)        # ramp up
    T[t2:t3] = T_tests[i]                        # steady state
    T[t3:t4] = T_tests[i] - T_approx(t3, t4, i)  # drop

    u[t1:t3] = u1_tests[i]  # const.

    t1 = int(t1 + t_seg)

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

dT = gradient_h4(T, dt)
dTf = interp1d(tvec, dT, kind='previous')  # piecewise, grab previous value

fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
fig.suptitle("Generated Data")
fig.supxlabel("Time (s)")

ax[0].set_ylabel("Heater Power (%)")
ax[0].plot(tvec, u)

ax[1].set_ylabel("Temperature (C)")
ax[1].plot(tvec, T)

ax[2].set_ylabel("Derivative (C/s)")
ax[2].plot(tvec, dT)

for i in range(len(ax)):
    ax[i].grid()

plt.show()

# =====================================================================
# =====================================================================
# GP model:

filename = os.path.join("models", "pyomo_tclab_v9.fokl")
try:
    GP = FoKLRoutines.load(filename)
except Exception as exception:
    GP = FoKLRoutines.FoKL(kernel=1, UserWarnings=False, aic=True)
    GP.fit([T - Tamb, u], dT, clean=True, pillow=[[0.01, 0.05], [0, 0]])
    GP.save(filename)

GP.coverage3(plot=True, title="Validation of GP", xlabel="Time (s)", ylabel="Derivative (C/s)")

# =====================================================================
# =====================================================================
# CONTROLLER REFERENCE TRAJECTORY:

# time grid
t0 = 0
tf = 999
dt = 1
n = round(tf / dt)
tvec = np.linspace(t0, tf, n + 1)

# ambient temperature
Tamb = 21.0

# time points of setpoint/reference values
tr = [t0, 50, 150, 450, 550, tf]

# setpoint/reference
def r(t):
    return np.interp(t, tr, np.array([Tamb, Tamb, 60, 60, 35, 35]) - Tamb)

# =====================================================================
# =====================================================================
# OPTIMIZATION USING GP MODEL - CREATION OF PYOMO MODEL:

# Create a Pyomo model
m = pyo.ConcreteModel('TCLab Heater with GP Model')

# Define time domain
m.t = dae.ContinuousSet(bounds=(t0, tf))

# Define the state variables as a function of time
m.Ts1 = pyo.Var(m.t)  # == T - Tamb

# Define the control variable (heater power) as a function of time
m.u1 = pyo.Var(m.t, bounds=(0, 100))

# Define the derivatives of the state variables
m.dTs1 = dae.DerivativeVar(m.Ts1, wrt=m.t)

# Fix the initial conditions
m.Ts1[t0].fix(0.0)

# Arguments to embed GP in Pyomo model:
xvars = [m.Ts1, m.u1]
yvar = m.dTs1
draws = 5

# Embed GP:
GP.to_pyomo(xvars, yvar, m, draws, with_blocks=False)

# =====================================================================
# =====================================================================
# U1 BENCHMARK:

u1_benchmark = np.loadtxt(os.path.join(dir, 'data', 'u1_benchmark_solution.csv'), delimiter=',')
u1_benchmark = np.concatenate([np.array([t0, 0])[np.newaxis, :],
                               u1_benchmark[1:107, :],
                               u1_benchmark[130:218, :],
                               u1_benchmark[243:354, :],
                               u1_benchmark[370:-1, :]], axis=0)  # remove oscillations

# =====================================================================
# =====================================================================
# OPTIMIZATION USING GP MODEL - SOLVE WITH IPOPT:

# Define the error model
@m.Integral(m.t)
def ise(m, t):
    return (r(t) - m.Ts1[t]) ** 2

# Define the objective function
@m.Objective(sense=pyo.minimize)
def objective(m):
    return m.ise

# Apply a collocation method to numerically integrate the differential equations
pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=100, wrt=m.t)

# Call our nonlinear optimization/equation solver, Ipopt
solver = pyo.SolverFactory('ipopt')
solver.options['linear_solver'] = 'ma57'
solver.solve(m, tee=True)

# Plot solution

tvec = m.t.data()

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle("IPOPT Results")
fig.supxlabel("Time (s)")

axs[0].plot()
axs[0].plot(tvec, r(tvec) + Tamb)
axs[0].plot(tvec, np.array(m.Ts1[:]()) + Tamb)
axs[0].legend(['r(t)', 'Ts1 (Var)'])

axs[1].plot(tvec, m.dTs1[:]())
axs[1].plot(tvec, GP.evaluate([m.Ts1[:](), m.u1[:]()], clean=True))  # GP confirmation
axs[1].legend(['dTs1 (Var)', 'dT (GP)'])

axs[2].plot(u1_benchmark[:, 0], u1_benchmark[:, 1])
axs[2].plot(tvec, m.u1[:]())
axs[2].legend(['u1 (benchmark)', 'u1 (Var)'])

for i in range(3):
    axs[i].grid()

plt.show()

