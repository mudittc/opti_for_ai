import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# Parameters
# -------------------------
M1, M2 = 1000, 100
k1, k2 = 20000, 15000
b1, b2 = 1500, 1000

K = np.array([4000, 1500, -2000, -400])  # control gain

# -------------------------
# External force
# -------------------------
def wind(t):
    return 300 + 200*np.sin(0.4*t)

# -------------------------
# System dynamics
# -------------------------
def model(t, x, nonlinear=False, active=False):

    x1, x2, x3, x4 = x
    w = wind(t)

    # control
    if active:
        u = -K @ x
        if nonlinear:
            u -= np.tanh(x2)   # nonlinear control
    else:
        u = 0

    # nonlinear damping
    nl = (x4 - x2)**3 if nonlinear else 0

    # equations
    dx1 = x2

    dx2 = (-(k1+k2)*x1 - (b1+b2)*x2 + k2*x3 + b2*x4 + w - u + nl) / M1

    dx3 = x4

    dx4 = (k2*x1 + b2*x2 - k2*x3 - b2*x4 + u - nl) / M2

    return [dx1, dx2, dx3, dx4]

# -------------------------
# Simulation
# -------------------------
t = np.linspace(0, 40, 2000)
x0 = [0, 0, 0, 0]

# run 4 cases
cases = {
    "Passive Linear": (False, False),
    "Active Linear": (False, True),
    "Passive Nonlinear": (True, False),
    "Active Nonlinear": (True, True)
}

results = {}

for name, (nl, act) in cases.items():
    sol = solve_ivp(model, [0, 40], x0, t_eval=t, args=(nl, act))
    results[name] = sol.y

# -------------------------
# Plot
# -------------------------
plt.figure()
for name in results:
    plt.plot(t, results[name][0], label=name)

plt.legend()
plt.title("Building Displacement")
plt.grid()
plt.show()