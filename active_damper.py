import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

# -------------------------
# Parameters
# -------------------------

M1, M2 = 1000, 100
k1, k2 = 20000, 15000
b1, b2 = 1500, 1000

# -------------------------
# State matrices
# -------------------------

A = np.array([
[0, 1, 0, 0],
[-(k1+k2)/M1, -(b1+b2)/M1,  k2/M1,  b2/M1],
[0, 0, 0, 1],
[ k2/M2, b2/M2, -k2/M2, -b2/M2]
])

Bw = np.array([[0],
               [1/M1],
               [0],
               [0]])

Bu = np.array([[0],
               [-1/M1],
               [0],
               [1/M2]])

# Combine inputs → [wind, control]
B = np.hstack((Bw, Bu))

# Output: choose what you care about
C = np.eye(4)
D = np.zeros((4,2))


K = np.array([4000, 1500, -2000, -400])

# Closed-loop system
A_cl = A - Bu @ K.reshape(1,4)
B_cl = Bw   # only wind remains external input

t = np.linspace(0, 40, 2000)

def wind(t):
    return 300 + 200*np.sin(0.4*t) + 50*np.sin(2*t)

w = wind(t)

sys = StateSpace(A_cl, B_cl, C, np.zeros((4,1)))

t_out, y, x = lsim(sys, U=w, T=t)

plt.figure()
plt.plot(t, y[:,0], label="Building displacement")
plt.plot(t, y[:,2], label="Damper displacement")
plt.legend()
plt.grid()
plt.title("Displacement (State-Space)")
plt.show()

plt.figure()
plt.plot(t, y[:,1], label="Building velocity")
plt.plot(t, y[:,3], label="Damper velocity")
plt.legend()
plt.grid()
plt.title("Velocity (State-Space)")
plt.show()
