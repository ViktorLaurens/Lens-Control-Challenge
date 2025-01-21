import numpy as np
import control as ctrl

# System parameters
m = 0.5  # mass in kg
g = 9.81  # gravity in m/s^2

# State-space matrices
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create state-space model
system = ctrl.ss(A, B, C, D)
print("State-space model:", system)



