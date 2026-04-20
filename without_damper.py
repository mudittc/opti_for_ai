import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

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
# WITH DAMPING SYSTEM
# -------------------------

A_damped = np.array([
[0, 1, 0, 0],
[-(k1+k2)/M1, -(b1+b2)/M1,  k2/M1,  b2/M1],
[0, 0, 0, 1],
[ k2/M2, b2/M2, -k2/M2, -b2/M2]
])

B = np.array([
[0],
[1/M1],
[0],
[0]
])

C = np.array([[1,0,0,0]])
D = np.array([[0]])

system_damped = StateSpace(A_damped,B,C,D)


# -------------------------
# WITHOUT DAMPING SYSTEM
# -------------------------

A_nodamp = np.array([
[0, 1, 0, 0],
[-(k1+k2)/M1, 0,  k2/M1,  0],
[0, 0, 0, 1],
[ k2/M2, 0, -k2/M2, 0]
])

system_nodamp = StateSpace(A_nodamp,B,C,D)


# -------------------------
# Simulation Time
# -------------------------

t = np.linspace(0,40,2000)


# -------------------------
# Wind Disturbance
# -------------------------

wind = 2e5*(1 - np.exp(-0.2*t)) + 5e4*np.sin(0.4*t)


# -------------------------
# Simulate Both Systems
# -------------------------

t, y1, x_damped = lsim(system_damped, U=wind, T=t)

t, y2, x_nodamp = lsim(system_nodamp, U=wind, T=t)


# -------------------------
# Plot Comparison
# -------------------------

plt.figure()

plt.plot(t, x_damped[:,0], label="With damping")
plt.plot(t, x_nodamp[:,0], label="Without damping")

plt.xlabel("Time")
plt.ylabel("Building Displacement")
plt.title("Building Response Comparison")
plt.legend()
plt.grid()

plt.show()