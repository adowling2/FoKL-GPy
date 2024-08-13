"""

Copied from ipynb so IPOPT can be run. To get csv of u1 benchmark solution.

"""

# import Pyomo library
import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import matplotlib.pyplot as plt

# parameters
alpha = 0.00016  # watts / (units P * percent U1)
P = 200  # P units
Ua = 0.050  # heat transfer coefficient from heater to environment
CpH = 2.2  # heat capacity of the heater (J/deg C)
CpS = 1.9  # heat capacity of the sensor (J/deg C)
Ub = 0.021  # heat transfer coefficient from heater to sensor
Tamb = 21.0  # ambient temperature

# setpoint/reference
def r(t):
    return np.interp(t, [0, 50, 150, 450, 550], [Tamb, Tamb, 60, 60, 35])

# Create a Pyomo model
m = pyo.ConcreteModel('TCLab Heater/Sensor')

# Define time domain
m.t = dae.ContinuousSet(bounds=(0, 1000))

# Define the state variables as a function of time
m.Th1 = pyo.Var(m.t)
m.Ts1 = pyo.Var(m.t)

# Define the derivatives of the state variables
m.dTh1 = dae.DerivativeVar(m.Th1)
m.dTs1 = dae.DerivativeVar(m.Ts1)

# Define the control variable (heater power) as a function of time
m.u1 = pyo.Var(m.t, bounds=(0, 100))


# Define the integral of the squared error
# Note: This modified objective tracks the sensor temperature error
# This is important because our GP model is trained on the sensor temperature
# We want to compare apples to apples
@m.Integral(m.t)
def ise(m, t):
    return (r(t) - m.Ts1[t]) ** 2


# Define the first differential equation
@m.Constraint(m.t)
def heater1(m, t):
    return (
            CpH * m.dTh1[t]
            == Ua * (Tamb - m.Th1[t]) + Ub * (m.Ts1[t] - m.Th1[t]) + alpha * P * m.u1[t]
    )


# Define the second differential equation
@m.Constraint(m.t)
def sensor1(m, t):
    return CpS * m.dTs1[t] == Ub * (m.Th1[t] - m.Ts1[t])


# Fix the initial conditions
m.Th1[0].fix(Tamb)
m.Ts1[0].fix(Tamb)


# Define the objective function
@m.Objective(sense=pyo.minimize)
def objective(m):
    return m.ise


# Apply a collocation method to numerically integrate the differential equations
pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=200, wrt=m.t)

# Call our nonlinear optimization/equation solver, Ipopt
pyo.SolverFactory('ipopt').solve(m)


def plot_results(m):
    """
    Plot results from Pyomo optimization

    Arguments:
    m: Pyomo model

    Returns:
    Nothing

    """

    # Plot the results
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(m.t, [m.Th1[t]() for t in m.t], label="Th1")
    ax[0].plot(m.t, [m.Ts1[t]() for t in m.t], label="Ts1")
    ax[0].legend()
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Temp. (°C)")
    ax[0].grid()

    ax[1].plot(m.t, [m.u1[t]() for t in m.t], label="U1")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("H. Power (%)")
    ax[1].grid()


plot_results(m)
plt.show()

u1_as_ndarray = np.zeros([len(m.t), 2])  # [t, u1]
i = -1
for t in m.t:
    i += 1
    u1_as_ndarray[i, :] = [t, m.u1[t]()]

np.savetxt("../data/u1_benchmark_solution.csv", u1_as_ndarray, delimiter=",")

