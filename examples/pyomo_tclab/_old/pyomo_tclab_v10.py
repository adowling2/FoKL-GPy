"""

Outline:
    - GP trained on artificial step tests data, yielding dT = f(T, u)
    - prediction of sine test as GP validation
    - for Pyomo model with increasing draws, e.g., [5, 10, 20, ..., ~1000],
        - IPOPT solution of r(t), with solution used to evaluate each draw in ODE

Acknowledgement:
    - https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/binaries.html#binary-packages
        - All technical papers, sales and publicity material resulting from use of the HSL codes within Ipopt must
        contain the following acknowledgement: HSL, a collection of Fortran codes for large-scale scientific
        computation. See http://www.hsl.rl.ac.uk.

"""
# =====================================================================
# =====================================================================
# MODULES:

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
import pandas as pd
import pyomo.environ as pyo
import pyomo.dae as dae

# =====================================================================
# =====================================================================
# SYSTEM-LEVEL PARAMETERS:

Tamb = 21           # ambient temperature (C)
Tmax = 75           # maximumum temperature, i.e., steady state of u=100; set s.t. u=50 yields ~55 C
t0 = 0              # start time (s)
dt = 1              # time step (s)
t_ramp = 200  # 1000       # estimated time to reach steady state (s)
t_rest = 20  # 50         # time to "rest" at steady state for sake of obtaining dT=0 training data
t_drop = 400  # 800        # estimated time to return to Tamb (s)
n = 20              # number of step tests to obtain data for

# =====================================================================
# =====================================================================
# DATA GENERATION:

u1_tests = np.linspace(100 / n, 100, n)  # heater power (%), const. values at which to obtain data
T_tests = Tamb + (Tmax - np.flip(np.logspace(np.log10(Tamb), np.log10(Tmax), n + 1))[1::])  # steady state temperature corresponding to u1_tests; decreases according to logscale as u increases

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

T += np.random.rand(len(T))  # add noise

def u_cutoff(u):
    """Enforce [0, 100] bounds."""
    u[u < 0] = 0
    u[u > 100] = 100
    return u

u = u_cutoff(u)

# =====================================================================
# =====================================================================
# SMOOTH AND DIFFERENTIATE:

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

T_raw = T
T = smooth(T_raw, window)  # smooth

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

dT = gradient_h4(T, dt)  # derivative of smoothed
dTf = interp1d(tvec, dT, kind='previous')  # piecewise, grab previous value

# fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
# fig.suptitle("Generated Data")
# fig.supxlabel("Time (s)")
#
# ax[0].set_ylabel("Heater Power (%)")
# ax[0].plot(tvec, u)
#
# ax[1].set_ylabel("Temperature (C)")
# ax[1].plot(tvec, T_raw)
# ax[1].plot(tvec, T)
# ax[1].legend(['Raw', 'Smooth'])
#
# ax[2].set_ylabel("Derivative (C/s)")
# ax[2].plot(tvec, dT)
#
# for i in range(len(ax)):
#     ax[i].grid()
#
# plt.show()

# =====================================================================
# =====================================================================
# TRAIN GP:

filename = os.path.join(dir, "models", "pyomo_tclab_v10.fokl")
try:
    GP = FoKLRoutines.load(filename)
except Exception as exception:
    GP = FoKLRoutines.FoKL(kernel=1, UserWarnings=False, aic=True)
    GP.fit([T - Tamb, u], dT, clean=True, pillow=[[0.01, 0.05], [0, 0]])
    GP.save(filename)

# Prepare Pyomo solver:
solver = pyo.SolverFactory('ipopt')
solver.options['linear_solver'] = 'ma57'

# # =====================================================================
# # =====================================================================
# # SINE TEST FOR GP VALIDATION:
#
# # =====================================================================
# # LOAD AND PARSE DATA (SINE TEST):
#
# sine_data = pd.read_csv(os.path.join(dir, "data", "tclab_sine_test.csv"))
#
# sine_tvec = sine_data["Time"].values
# sine_u = sine_data["Q1"].values
# sine_T = sine_data["T1"].values
# sine_Tamb = sine_T[0]
#
# sine_T_raw = sine_T
# sine_T = smooth(sine_T_raw, window)  # smooth
#
# sine_dT = gradient_h4(sine_T, sine_tvec[1] - sine_tvec[0])  # derivative
# sine_u = u_cutoff(sine_u)
#
# # =====================================================================
# # CONTROLLER REFERENCE TRAJECTORY (SINE TEST):
#
# def sine_r(t):
#     return np.interp(t, sine_tvec, sine_T - sine_Tamb)
#
# # =====================================================================
# # PYOMO MODEL (SINE TEST):
#
# sine_m = pyo.ConcreteModel("Sine Test")
# sine_m.t = dae.ContinuousSet(bounds=(sine_tvec[0], sine_tvec[-1]))
# sine_m.TmTamb = pyo.Var(sine_m.t, bounds=GP.minmax[0])  # == T - Tamb
# sine_m.u = pyo.Var(sine_m.t, bounds=(0, 100))
# sine_m.dT = dae.DerivativeVar(sine_m.TmTamb, wrt=sine_m.t)
# sine_m.TmTamb[sine_tvec[0]].fix(0.0)
# GP.to_pyomo([sine_m.TmTamb, sine_m.u], sine_m.dT, sine_m, 20, with_blocks=False)
#
# # =====================================================================
# # IPOPT (SINE TEST):
#
# # Define the error model
# @sine_m.Integral(sine_m.t)
# def ise(sine_m, t):
#     return (sine_r(t) - sine_m.TmTamb[t]) ** 2
#
# # Define the objective function
# @sine_m.Objective(sense=pyo.minimize)
# def objective(sine_m):
#     return sine_m.ise
#
# # Apply a collocation method to numerically integrate the differential equations
# pyo.TransformationFactory('dae.collocation').apply_to(sine_m, nfe=len(sine_tvec), wrt=sine_m.t)
#
# # Call our nonlinear optimization/equation solver, Ipopt
# solver.solve(sine_m, tee=True)
#
# # Plot solution
#
# sine_sol_tvec = sine_m.t.data()
#
# fig, axs = plt.subplots(3, sharex=True)
# fig.suptitle("IPOPT Results of Sine Test for GP Validation")
# fig.supxlabel("Time (s)")
#
# axs[0].plot()
# axs[0].plot(sine_tvec, sine_T)
# axs[0].plot(sine_sol_tvec, np.array(sine_m.TmTamb[:]()) + sine_Tamb)
# axs[0].legend(['T (data)', 'T (Pyomo)'])
#
# axs[1].plot(sine_tvec, sine_dT)
# axs[1].plot(sine_sol_tvec, sine_m.dT[:]())
# axs[1].legend(['dT (data)', 'dT (Pyomo)'])
#
# axs[2].plot(sine_tvec, sine_u)
# axs[2].plot(sine_sol_tvec, sine_m.u[:]())
# axs[2].legend(['u (data)', 'u (Pyomo)'])
#
# for i in range(3):
#     axs[i].grid()
#
# plt.show()

# =====================================================================
# =====================================================================
# TARGET:

# =====================================================================
# CONTROLLER REFERENCE TRAJECTORY:

# time grid
r_t0 = 0
r_tf = 999
r_dt = 1
r_n = round(r_tf / r_dt)
r_tvec = np.linspace(r_t0, r_tf, r_n + 1)

# ambient temperature
r_Tamb = 21.0

# time points of setpoint/reference values
r_t = [r_t0, 50, 150, 450, 550, r_tf]

# setpoint/reference
def r(t):
    return np.interp(t, r_t, np.array([r_Tamb, r_Tamb, 60, 60, 35, 35]) - r_Tamb)

# derivative of setpoint/reference
dr_interp = interp1d(r_t, [0, (60 - r_Tamb) / 100, 0, (35 - 60) / 100, 0, 0], kind='previous')
def dr(t):
    if isinstance(t, float) or isinstance(t, int):
        return float(dr_interp(t))
    else:  # list or ndarray
        return np.array(dr_interp(t))

# =====================================================================
# U1 BENCHMARK:

u1_benchmark = np.loadtxt(os.path.join(dir, 'data', 'u1_benchmark_solution.csv'), delimiter=',')
u1_benchmark = np.concatenate([np.array([t0, 0])[np.newaxis, :],
                               u1_benchmark[1:107, :],
                               u1_benchmark[130:218, :],
                               u1_benchmark[243:354, :],
                               u1_benchmark[370:-1, :]], axis=0)  # remove oscillations

# =====================================================================
# OPTIMIZATION USING GP MODEL:

draws_list = (np.linspace(10, 90, 9, dtype=int).tolist() +
              np.linspace(100, 1000, 10, dtype=int).tolist())
u_solution = pd.DataFrame(columns=["Time"] + draws_list)

for draws in draws_list:
    print(f"Draws = {draws}")

    # =================================================================
    # PYOMO MODEL:
    
    m = pyo.ConcreteModel("TCLab Heater with GP Model")
    m.t = dae.ContinuousSet(bounds=(r_t0, r_tf))
    m.TmTamb = pyo.Var(m.t, bounds=GP.minmax[0])  # == T - Tamb
    m.u = pyo.Var(m.t, bounds=(0, 100))
    m.dT = dae.DerivativeVar(m.TmTamb, wrt=m.t)
    m.TmTamb[r_t0].fix(0.0)
    GP.to_pyomo([m.TmTamb, m.u], m.dT, m, draws, with_blocks=False)

    # =================================================================
    # IPOPT:

    # Define the error model
    @m.Integral(m.t)
    def ise(m, t):
        return (r(t) - m.TmTamb[t]) ** 2

    # Define the objective function
    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return m.ise

    # Apply a collocation method to numerically integrate the differential equations
    pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=len(r_tvec), wrt=m.t)

    # Call our nonlinear optimization/equation solver, Ipopt
    solver.solve(m, tee=True)

    # =================================================================
    # SOLUTION:

    if draws == draws_list[0]:
        u_solution["Time"] = m.t.data()  # same for all draws
    u_solution[draws] = m.u[:]()  # save solution to DataFrame

    u_solution.to_csv(os.path.join("data", "pyomo_tclab_v10_ipopt_u.csv"))  # save DataFrame to csv

    # # =================================================================
    # # PLOT:
    #
    # plt_tvec = m.t.data()
    #
    # fig, axs = plt.subplots(3, sharex=True)
    # fig.suptitle(f"IPOPT Results, {draws} draws")
    # fig.supxlabel("Time (s)")
    #
    # axs[0].plot(plt_tvec, r(plt_tvec) + r_Tamb)
    # axs[0].plot(plt_tvec, np.array(m.TmTamb[:]()) + r_Tamb)
    # axs[0].legend(['r(t)', 'T (Pyomo)'])
    #
    # axs[1].plot(plt_tvec, dr(plt_tvec))
    # axs[1].plot(plt_tvec, m.dT[:]())
    # axs[1].legend(['d(r)/dt', 'dT (Pyomo)'])
    #
    # axs[2].plot(u1_benchmark[:, 0], u1_benchmark[:, 1])
    # axs[2].plot(plt_tvec, m.u[:]())
    # axs[2].legend(['u (benchmark)', 'u (Pyomo)'])
    #
    # for i in range(3):
    #     axs[i].grid()
    #
    # plt.show()

