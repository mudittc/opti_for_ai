import numpy as np
import matplotlib.pyplot as plt

# Time steps
T = 100

# State dimension = 2
n = 2

# Observation dimension = 1
m = 1

# State transition matrix
A = np.array([[1, 1],
              [0, 1]])

# Observation matrix
C = np.array([[1, 0]])

# Process noise covariance
Q = np.array([[0.1, 0],
              [0, 0.1]])

# Measurement noise covariance
R = np.array([[1]])

# Initial state
x = np.array([[0],
              [1]])  # start at position 0, velocity 1

# Storage
states = []
observations = []

for k in range(T):
    # Process noise
    w = np.random.multivariate_normal(mean=[0,0], cov=Q).reshape(-1,1)
    
    # State update
    x = A @ x + w
    
    # Measurement noise
    v = np.random.normal(0, np.sqrt(R[0,0]), size=(1,1))
    
    # Observation
    y = C @ x + v
    
    states.append(x.flatten())
    observations.append(y.flatten())

states = np.array(states)
observations = np.array(observations)

# Plot
plt.figure()
plt.plot(states[:,0], label="True Position")
plt.plot(observations[:,0], label="Observed Position")
plt.legend()
plt.show()