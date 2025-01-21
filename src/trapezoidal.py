import numpy as np
import matplotlib.pyplot as plt

# Define the trapezoidal profile function
def trapezoidal_profile(t, t_start, t_end, q_start, q_end, acc_time, dec_time):
    total_time = t_end - t_start
    constant_vel_time = total_time - acc_time - dec_time
    max_vel = (q_end - q_start) / (constant_vel_time + 0.5 * acc_time + 0.5 * dec_time)
    acc = max_vel / acc_time

    if t < t_start:
        q = q_start
        dq = 0
        ddq = 0
    elif t < t_start + acc_time:  # Acceleration phase
        q = q_start + 0.5 * acc * (t - t_start)**2
        dq = acc * (t - t_start)
        ddq = acc
    elif t < t_start + acc_time + constant_vel_time:  # Constant velocity phase
        q = q_start + 0.5 * acc * acc_time**2 + max_vel * (t - (t_start + acc_time))
        dq = max_vel
        ddq = 0
    elif t < t_end - dec_time:  # Constant velocity phase
        q = q_start + 0.5 * acc * acc_time**2 + max_vel * constant_vel_time
        dq = max_vel
        ddq = 0
    elif t < t_end:  # Deceleration phase
        time_dec = t - (t_end - dec_time)
        q = q_end - 0.5 * acc * (dec_time - time_dec)**2
        dq = max_vel - acc * time_dec
        ddq = -acc
    else:
        q = q_end
        dq = 0
        ddq = 0

    return q, dq, ddq

# Time settings
T_wait = 0.2  # Waiting time at A and B
T_move = 0.2  # Time to move between points A and B
total_cycle_time = 2 * (T_wait + T_move)

t_total = np.linspace(0, total_cycle_time, 1000)
position = []
velocity = []
acceleration = []

# Trajectory for moving from A to B and back to A
q_A, q_B = 0, 1  # Positions of A and B

for t in t_total:
    if t < T_wait:
        q, dq, ddq = q_A, 0, 0  # Waiting at A
    elif t < T_wait + T_move:
        q, dq, ddq = trapezoidal_profile(t, T_wait, T_wait + T_move, q_A, q_B, 0.05, 0.05)
    elif t < 2 * T_wait + T_move:
        q, dq, ddq = q_B, 0, 0  # Waiting at B
    else:
        q, dq, ddq = trapezoidal_profile(t, 2 * T_wait + T_move, total_cycle_time, q_B, q_A, 0.05, 0.05)

    position.append(q)
    velocity.append(dq)
    acceleration.append(ddq)

# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t_total, position, label="Position (m)")
plt.ylabel("Position")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_total, velocity, label="Velocity (m/s)")
plt.ylabel("Velocity")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_total, acceleration, label="Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()
plt.grid()

plt.suptitle("Trapezoidal Motion Profiles")
plt.show()