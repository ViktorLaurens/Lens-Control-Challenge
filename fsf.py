import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from reference import t_total_extended, position_extended, velocity_extended, acceleration_extended
from model import A, B, C, m
t_ref = t_total_extended
pos_ref = position_extended
vel_ref = velocity_extended
acc_ref = acceleration_extended

# Desired closed-loop poles (choose for stability and performance)
desired_poles = np.array([-20, -25])  # Place poles for the closed-loop system

# Compute state feedback gain using pole placement
K = ctrl.place(A, B, desired_poles)
print("State feedback gain K:", K)

# Initial conditions
x0 = np.array([0, 0])  # Initial position and velocity

# Define the closed-loop system dynamics function with feedforward term
def closed_loop_dynamics_with_ff(t, x):
    # Find the closest reference index
    idx = np.argmin(np.abs(t_ref - t))

    # Get reference values for position, velocity, and acceleration
    q_ref = pos_ref[idx]
    dq_ref = vel_ref[idx]
    ddq_ref = acc_ref[idx]

    # Compute feedforward control
    u_ff = m * ddq_ref  # Feedforward based on reference acceleration
    # u_ff = 0 # No feedforward control

    # Compute feedback control
    x_ref = np.array([[q_ref], [dq_ref]])
    u_fb = -K @ (x.reshape(-1, 1) - x_ref)

    # Total control input
    u = u_ff + u_fb

    dxdt = A @ x.reshape(-1, 1) + B @ u
    return dxdt.flatten()  # Return as 1D array for solve_ivp

# Define the closed-loop system dynamics function without feedforward term
def closed_loop_dynamics_without_ff(t, x):
    # Find the closest reference index
    idx = np.argmin(np.abs(t_ref - t))

    # Get reference values for position, velocity, and acceleration
    q_ref = pos_ref[idx]
    dq_ref = vel_ref[idx]

    # Compute feedback control
    x_ref = np.array([[q_ref], [dq_ref]])
    u = -K @ (x.reshape(-1, 1) - x_ref)

    dxdt = A @ x.reshape(-1, 1) + B @ u
    return dxdt.flatten()  # Return as 1D array for solve_ivp

# Solve the closed-loop system using numerical integration
fsf_solution_with_ff = solve_ivp(closed_loop_dynamics_with_ff, [t_ref[0], t_ref[-1]], x0, t_eval=t_ref)
fsf_solution_without_ff = solve_ivp(closed_loop_dynamics_without_ff, [t_ref[0], t_ref[-1]], x0, t_eval=t_ref)

# Plot the reference and both controlled motions with different colors and legend

plt.figure(figsize=(10, 8))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(t_ref, pos_ref, 'k--', label="Reference Position")
plt.plot(t_ref, fsf_solution_with_ff.y[0], 'b-', label="Position with Feedforward")
plt.plot(t_ref, fsf_solution_without_ff.y[0], 'r-', label="Position without Feedforward")
plt.ylabel("Position (m)")
plt.title("Position Tracking")
plt.legend()
plt.grid()

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(t_ref, vel_ref, 'k--', label="Reference Velocity")
plt.plot(t_ref, fsf_solution_with_ff.y[1], 'b-', label="Velocity with Feedforward")
plt.plot(t_ref, fsf_solution_without_ff.y[1], 'r-', label="Velocity without Feedforward")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Tracking")
plt.legend()
plt.grid()

plt.suptitle("Comparison of Closed-Loop Response with and without Feedforward")
plt.show()
