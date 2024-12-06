import numpy as np
import matplotlib.pyplot as plt
from helper_functions import Rzyx


def body_to_global(positions, euler_angles):
    """
    Convert positions from body frame to global frame using Euler angles.
    """
    global_positions = []
    for pos, angles in zip(positions, euler_angles):
        phi, theta, psi = angles
        R = Rzyx(phi, theta, psi)
        global_pos = R @ pos
        global_positions.append(global_pos)
    return global_positions


# Load Data
run_history = np.load('run_history_27.npy')
timestep = 1/60 # this is hardcoded, need to figure out how to save it in other file and pass in here

# Set Up Plots
time = np.arange(0, len(run_history[0])*timestep/16, timestep)
fig = plt.figure()
fig.suptitle('Simulation using Estimated Parameters')

ax1 = fig.add_subplot(131)
ax1.set_xlabel('time')
ax1.set_ylabel('X Position')

ax2 = fig.add_subplot(132)
ax2.set_xlabel('time')
ax2.set_ylabel('Y Position')

ax3 = fig.add_subplot(133)
ax3.set_xlabel('time')
ax3.set_ylabel('Z Position')

# Plot True States
true_states = run_history[0].reshape(-1, 16)
positions = [state[:3] for state in true_states]
euler_angles = [state[3:6] for state in true_states]
global_positions = body_to_global(positions, euler_angles)
ax1.plot(time, [pos[0] for pos in global_positions], label='True', color='black', linewidth=2)
ax2.plot(time, [pos[1] for pos in global_positions], label='True', color='black', linewidth=2)
ax3.plot(time, [pos[2] for pos in global_positions], label='True', color='black', linewidth=2)

# Plot Iterative States
for i, run in enumerate(run_history[1:-1]):
    # if i%200 == 0:
    if i==10 or i== 500 or i==1000 or i==5000 or i==10000:
        run = run.reshape(-1, 16)
        
        positions = [state[:3] for state in run]
        euler_angles = [state[3:6] for state in run]
        global_positions = body_to_global(positions, euler_angles)
        ax1.plot(time, [pos[0] for pos in global_positions], label=f'Run {i}')
        ax2.plot(time, [pos[1] for pos in global_positions], label=f'Run {i}')
        ax3.plot(time, [pos[2] for pos in global_positions], label=f'Run {i}')

# Plot Estimated States
estimated_states = run_history[-1].reshape(-1, 16)
positions = [state[:3] for state in estimated_states]
euler_angles = [state[3:6] for state in estimated_states]
global_positions = body_to_global(positions, euler_angles)
ax1.plot(time, [pos[0] for pos in global_positions], label='Estimated', color='red', linestyle='--')
ax2.plot(time, [pos[1] for pos in global_positions], label='Estimated', color='red', linestyle='--')
ax3.plot(time, [pos[2] for pos in global_positions], label='Estimated', color='red', linestyle='--')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')
# plt.tight_layout()
plt.show()

