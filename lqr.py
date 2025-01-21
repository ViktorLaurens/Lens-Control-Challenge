import numpy as np
import control as ctrl
from model import A, B

# Define cost matrices
Q = np.array([[1, 0], [0, 1]])  # Penalizes position and velocity errors
R = np.array([[0.1]])  # Penalizes control effort

# Compute the optimal gain
K, _, _ = ctrl.lqr(A, B, Q, R)
print("State feedback gain K:", K)