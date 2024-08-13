"""
Part B:
    - load training data
    - train GP
    - setup Pyomo model
    - solve Pyomo model
    - save IPOPT solution
"""
import os
dir = os.path.abspath('')  # directory of notebook
import sys
sys.path.append(os.path.join(dir, '..', '..'))  # package directory
from src.FoKL import FoKLRoutines
from src.FoKL.fokl_to_pyomo import fokl_to_pyomo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
import pyomo.dae as dae


def smooth(x, window):
    """Apply centered average of size window."""
    x_smooth = np.zeros_like(x)
    w2 = int(np.floor(window / 2))
    w2p1 = w2 + 1

    # bleed in:
    for i in range(w2):
        x_smooth[i] = np.mean(x[:(i + w2p1)])

    # center:
    for i in range(w2, x_smooth.size - w2):
        x_smooth[i] = np.mean(x[(i - w2):(i + w2p1)])

    # bleed out:
    for i in range(-w2, 0):
        x_smooth[i] = np.mean(x[(i - w2)::])

    return x_smooth


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


def main(INDEX_SCENARIOS):
    # Load data:
    data = pd.read_csv(os.path.join(dir, "data", "v11_training_data.csv"))
    tvec = data["t"]
    T = smooth(data["T"], 9)  # smooth
    T = T - T[0]  # subtract Tamb
    u = data["u"]
    dt = tvec[1] - tvec[0]
    dT = gradient_h4(T, dt)  # derivative of smoothed

    # Train GP:
    filename = os.path.join(dir, "models", "v11.fokl")
    try:
        GP = FoKLRoutines.load(filename)
    except Exception as exception:
        GP = FoKLRoutines.FoKL(kernel=1, UserWarnings=False, aic=True)
        GP.fit([T, u], dT, clean=True, pillow=[[0.01, 0.05], [0, 0]])
        GP.save(filename)

    # =====================================================================
    # CONTROLLER REFERENCE TRAJECTORY:

    t0 = 0                            # start time
    tf = 999                          # end time
    Tamb = 21.0                       # ambient temperature
    tr = [t0, 50, 150, 450, 550, tf]  # time points of setpoint/reference values

    def r(t):  # setpoint/reference (wrt Tamb)
        return np.interp(t, tr, np.array([Tamb, Tamb, 60, 60, 35, 35]) - Tamb)

    # =================================================================
    # PYOMO MODEL:

    draws = 5

    m = pyo.ConcreteModel("TCLab Heater with GP Model")
    m.t = dae.ContinuousSet(bounds=(t0, tf))
    m.u = pyo.Var(m.t, bounds=(0, 100))

    if INDEX_SCENARIOS:

        m.s = pyo.Set(initialize=range(draws))  # scenarios
        m.T = pyo.Var(m.t, m.s, bounds=GP.minmax[0])  # == T - Tamb
        m.dT = dae.DerivativeVar(m.T, wrt=m.t)

        for s in m.s:
            m.T[t0, s].fix(0.0)  # initial condition
        m.u[t0].fix(0.0)

        fokl_to_pyomo(GP, [m.T, m.u], m.dT, m, m.t, m.s)

        # Integral of squared error:
        @m.Integral(m.t, m.s, wrt=m.t)
        def ise(m, t, s):
            return (r(t) - m.T[t, s]) ** 2

        # Define the objective function:
        @m.Objective(sense=pyo.minimize)
        def objective(m):
            return pyo.summation(m.ise)  # == sum(m.ise[s] for s in m.s)

        m.GP0_constr.deactivate()
        m.GP0_constr[:, 0].activate()  # activate first scenario only, as not all constraints can be satisfied

    else:

        m.T = pyo.Var(m.t, bounds=GP.minmax[0])  # == T - Tamb
        m.dT = dae.DerivativeVar(m.T, wrt=m.t)

        m.T[t0].fix(0.0)  # initial condition
        m.u[t0].fix(0.0)

        fokl_to_pyomo(GP, [m.T, m.u], m.dT, m, m.t)

        @m.Integral(m.t, wrt=m.t)
        def ise(m, t):
            return (r(t) - m.T[t]) ** 2

        @m.Objective(sense=pyo.minimize)
        def objective(m):
            return m.ise

    # =================================================================
    # SOLUTION:

    # Prepare Pyomo solver:
    solver = pyo.SolverFactory('ipopt')
    # solver.options['linear_solver'] = 'ma57'

    # Apply a collocation method to numerically integrate the differential equations
    pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=100, wrt=m.t)

    # Call our nonlinear optimization/equation solver, Ipopt
    solver.solve(m, tee=True)

    # Plot and save solution:
    plt.figure()
    if INDEX_SCENARIOS:
        for s in [0]:  # m.s:
            plt.plot(np.array(m.t), np.array(m.T[:, s]()))
    else:
        plt.plot(np.array(m.t), np.array(m.T[:]()))
    plt.plot(np.array(m.t), np.array(m.u[:]()))
    plt.show()

    return


if __name__ == '__main__':
    INDEX_SCENARIOS = True
    main(INDEX_SCENARIOS)

