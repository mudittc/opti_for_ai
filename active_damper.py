import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# System Parameters
# -------------------------

M1 = 1000
M2 = 100

k1 = 20000
k2 = 15000

b1 = 1500
b2 = 1000

# -------------------------
# State Matrix
# -------------------------

A = np.array([
[0, 1, 0, 0],
[-(k1+k2)/M1, -(b1+b2)/M1,  k2/M1,  b2/M1],
[0, 0, 0, 1],
[ k2/M2, b2/M2, -k2/M2, -b2/M2]
])

# wind disturbance input
Bw = np.array([[0],
               [1/M1],
               [0],
               [0]])

# actuator force input
Bu = np.array([[0],
               [-1/M1],
               [0],
               [1/M2]])

# -------------------------
# Controller Gain
# -------------------------

K = np.array([4000, 1500, -2000, -400])

# -------------------------
# Time
# -------------------------

t = np.linspace(0,40,2000)

# -------------------------
# Wind Function
# -------------------------

def wind(t):
    return 300 + 200*np.sin(0.4*t) + 50*np.sin(2*t)

# -------------------------
# System Dynamics
# -------------------------

def dynamics(t, x):

    w = wind(t)

    u = -K @ x     # active control force

    dx = A @ x + Bw.flatten()*w + Bu.flatten()*u

    return dx

# -------------------------
# Simulation
# -------------------------

sol = solve_ivp(dynamics,[0,40],[0,0,0,0],t_eval=t)

x = sol.y.T

# -------------------------
# Plot Displacement
# -------------------------

plt.figure()

plt.plot(t,x[:,0],label="Building displacement")
plt.plot(t,x[:,2],label="Damper displacement")

plt.xlabel("Time")
plt.ylabel("Displacement")
plt.title("Active Mass Damper Simulation")

plt.legend()
plt.grid()

plt.show()

# -------------------------
# Plot Velocity
# -------------------------

plt.figure()

plt.plot(t,x[:,1],label="Building velocity")
plt.plot(t,x[:,3],label="Damper velocity")

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity Response")

plt.legend()
plt.grid()

plt.show()

# active damper 