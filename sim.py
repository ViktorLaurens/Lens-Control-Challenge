import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from model import A, B, C, m, g, system
from lqr import K
import control as ctrl

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




# PD control gains
Kp = 100
Kd = 20

def pd_control_dynamics(t, x):
    q_ref = 0  # Desired position
    dq_ref = 0  # Desired velocity
    ddq_ref = 0  # Desired acceleration
    
    # PD control law with feedforward and gravity compensation
    u = Kp * (q_ref - x[0]) + Kd * (dq_ref - x[1]) + m * ddq_ref + m * g
    
    dxdt = [x[1], u / m - g]  # Second-order dynamics
    return dxdt

# Simulate PD control
solution_pd = solve_ivp(pd_control_dynamics, [0, 10], x0, t_eval=t_span)

# Plot PD results
plt.plot(t_span, solution_pd.y[0], label="Position (PD Control)")
plt.plot(t_span, solution_pd.y[1], label="Velocity (PD Control)")
plt.title("PD Control Response")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid()
plt.show()




# Process noise covariance (Q) and measurement noise covariance (R)
Q_kf = np.array([[0.1]])  # Process noise covariance
R_kf = np.array([[0.01]])  # Measurement noise covariance

# Compute the optimal observer gain L using LQE
L, _, _ = ctrl.lqe(system, Q_kf, R_kf)
print("Observer gain L:", L)

# Define simulation time
t_span = np.linspace(0, 10, 1000)

# Initial conditions
x0 = np.array([1, 0])  # Initial state [position, velocity]
x_hat0 = np.array([0, 0])  # Initial state estimate

def true_system(t, x):
    u = np.array([0])  # No external input
    print("x shape:", x.shape)  # Should print (2,)
    print("u shape:", u.shape)  # Should print (1,)
    dxdt = A @ x + B @ u
    print("dxdt shape:", dxdt.shape)  # Should print (2,)
    return dxdt

true_sol = solve_ivp(true_system, [t_span[0], t_span[-1]], x0, t_eval=t_span)
measurements = true_sol.y[0] + np.random.normal(0, 0.1, len(t_span))

# Observer simulation
x_hat = x_hat0.reshape(-1, 1)
x_estimates = []

for k in range(len(t_span)):
    y_measured = measurements[k]
    x_hat = (A @ x_hat + L @ (y_measured - C @ x_hat))
    x_estimates.append(x_hat.flatten()[0])

# Plot results
plt.plot(t_span, measurements, label="Measured Position")
plt.plot(t_span, true_sol.y[0], label="True Position")
plt.plot(t_span, x_estimates, label="Estimated Position (LQE)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("LQE State Estimation")
plt.legend()
plt.grid()
plt.show()

