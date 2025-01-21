import numpy as np
import matplotlib.pyplot as plt
from trapezoidal import trapezoidal_profile, T_wait, T_move, total_cycle_time, q_A, q_B, t_total

# Number of cycles to repeat the motion
num_cycles = 3

# Extend the total simulation time
t_total_extended = np.linspace(0, total_cycle_time * num_cycles, 1000 * num_cycles)

# Extended arrays for position, velocity, and acceleration
position_extended = []
velocity_extended = []
acceleration_extended = []

for cycle in range(num_cycles):
    for t in t_total:
        t_shifted = t + cycle * total_cycle_time
        if t < T_wait:
            q, dq, ddq = q_A, 0, 0  # Waiting at A
        elif t < T_wait + T_move:
            q, dq, ddq = trapezoidal_profile(t, T_wait, T_wait + T_move, q_A, q_B, 0.05, 0.05)
        elif t < 2 * T_wait + T_move:
            q, dq, ddq = q_B, 0, 0  # Waiting at B
        else:
            q, dq, ddq = trapezoidal_profile(t, 2 * T_wait + T_move, total_cycle_time, q_B, q_A, 0.05, 0.05)

        position_extended.append(q)
        velocity_extended.append(dq)
        acceleration_extended.append(ddq)

# Plot the results for extended reference trajectory
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t_total_extended, position_extended, label="Position (m)")
plt.ylabel("Position")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_total_extended, velocity_extended, label="Velocity (m/s)")
plt.ylabel("Velocity")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_total_extended, acceleration_extended, label="Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()
plt.grid()

plt.suptitle("Trapezoidal Motion Profiles - 3 Cycles")
plt.show()
