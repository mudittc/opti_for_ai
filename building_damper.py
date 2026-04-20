import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

# -------------------------
# System Parameters
# -------------------------

M1 = 2e7      # 20,000 tons building, building mass
M2 = 4e5      # 400 ton damper

k1 = 8e7      #building stiffness
k2 = 1.5e7    #couplign stiffness

b1 = 3e5      #building dampning
b2 = 1e5      #damper damping


# -------------------------
# State Space Matrices
# -------------------------

A = np.array([
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

system = StateSpace(A,B,C,D)

# -------------------------
# Simulation Time
# -------------------------

t = np.linspace(0,40,2000)

# -------------------------
# Wind Disturbance
# -------------------------

wind = 2e5*(1 - np.exp(-0.2*t)) + 5e4*np.sin(0.4*t)

# -------------------------
# Simulate System
# -------------------------

t, y, x = lsim(system, U=wind, T=t)

# -------------------------
# Plot Displacements
# -------------------------

plt.figure()

plt.plot(t, x[:,0], label="Building displacement (x1)")
plt.plot(t, x[:,2], label="Damper displacement (x2)")

plt.xlabel("Time")
plt.ylabel("Displacement")
plt.title("Skyscraper Tuned Mass Damper Simulation")
plt.legend()
plt.grid()

plt.show()


# -------------------------
# Plot Velocities
# -------------------------

plt.figure()

plt.plot(t, x[:,1], label="Building velocity (v1)")
plt.plot(t, x[:,3], label="Damper velocity (v2)")

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity Response")
plt.legend()
plt.grid()

plt.show()


# non active damper