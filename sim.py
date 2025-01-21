import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from model import A, B
from lqr import K

# Define closed-loop system dynamics
def closed_loop_dynamics(t, x):
    u = -K @ x  # Control law
    dxdt = A @ x + B @ u
    return dxdt.flatten()

# Initial state
x0 = [1, 0]  # Initial displacement and velocity

# Time span
t_span = np.linspace(0, 10, 1000)

# Solve the system using numerical integration
solution = solve_ivp(closed_loop_dynamics, [t_span[0], t_span[-1]], x0, t_eval=t_span)

# Plot results
plt.plot(t_span, solution.y[0], label="Position (m)")
plt.plot(t_span, solution.y[1], label="Velocity (m/s)")
plt.title("Closed-Loop System Response with LQR")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid()
plt.show()
