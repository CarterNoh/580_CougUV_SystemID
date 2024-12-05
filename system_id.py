import numpy as np
import scipy.optimize as opt
from coug import Coug
from helper_functions import Rzyx
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

######## CougUV State Estimation #########
run_history = []

def generate_commands(u_semantic: list, timestep) -> np.ndarray:
    '''
    :param u_semantic: A list of duples, where the first item in the duple is a command and the second item is how long to hold that command.
    
    Four control inputs: Rudder angle, left elevator angle, right elevator angle, and thruster rpm. 
    Takes input of a command and desired timespan, and generates a Nx4 matrix, where each row is the commands at that timestep. 
    '''
    commands = []
    for command, duration in u_semantic:
        assert len(command) == 4, "Command must be a list of 4 values"
        assert all(isinstance(x, (int, float)) for x in command), "Command values must be integers or floats"

        steps = int(duration/timestep)
        for _ in range(steps):
            commands.append(command)
    commands = np.array(commands)

    return commands

def simulate(params, u, timestep):

    if len(params) == 0:
        param_dict = None
    else:
        param_dict = {
            # Environment Parameters
            # 'rho'       : params[0],
            'V_c'       : params[0],
            'beta_c'    : params[1],

            # Physical Parameters
            'r_bg'          : params[2:5],
            # 'r_bb'          : params[3:6],
            # 'm'             : params[3],
            # 'L'             : params[3],
            'diam'          : params[5],
            'area_fraction' : params[6],

            # Damping Parameters: 
            'T_surge'       : params[7],
            'T_sway'        : params[8],
            'T_heave'       : params[9],
            'T_yaw'         : params[10],
            'zeta_roll'     : params[11],
            'zeta_pitch'    : params[12],
            'Cd'            : params[13],
            'e'             : params[14],
            'r44'           : params[15],

            # Fin Parameters
            'S_fin'         : params[16],
            'x_fin'         : params[17],
            'fin_center'    : params[18],
            'CL_delta_r'    : params[19],
            # 'CL_delta_s'    : params[18],
            'T_delta'       : params[20],

            # Motor Parameters
            # 'D_prop'        : params[16],
            # 't_prop'        : params[16],
            # 'Ja_max'        : params[16],
            # 'Va'            : params[16],
            # 'KT_0'          : params[],
            # 'KQ_0'          : params[],
            # 'KT_max'        : params[],
            # 'KQ_max'        : params[],
            'T_n'           : params[21],
            }

    coug = Coug(param_dict)
    states = []

    for command in u:
        coug.step(command, timestep)
        state = np.concatenate((coug.nu.copy(), coug.eta.copy(), coug.u_actual.copy()), axis=0)
        states.append(state)

    # Output: A matrix of the true state at each timestep, flattened into a vector
    states = np.array(states).flatten()

    # Save the states to external variable for plotting
    run_history.append(states)

    return states

def residuals(params, truth, u, timestep):
    '''
    Residuals for the optimizer to minimize.
    '''

    return truth - simulate(params, u, timestep)

def cost(params, truth, u, timestep):
    '''
    Cost function for the optimizer to minimize. 
    '''
    return np.linalg.norm(residuals(params, truth, u, timestep))





########## MAIN ##########
timestep = 1/60    # (s). Set to 60 fps, what holoocean commonly does.

# Create list of commands
semantic_commands = [([10, 0, 0, 2000], .3),
                     ([ 0, 0, 0, 0   ], .1),
                     ([ 0,10,10, 1000], .3),
                     ([ 0,-5,-5, 1000], .3),
                     ([ 0, 0, 0, 2000], .3),
                     ([20,20,-20,1000], .3),
                     ]

commands = generate_commands(semantic_commands, timestep)

# Simulate with true parameters to get ground truth
true_states = simulate([], commands, timestep)

# Initialize parameters 
params_init = [
    # Environment Parameters
    # 800, # rho
    1, # V_c
    np.pi*3/4, # beta_c: only identifiable if V_c =/= 0?

    # Physical Parameters
    .1, .1, .1, # r_bg
    # .1, .1, .1, # r_bb
    # 15, # m
    # 1.2, # L
    1, # diam
    0.5, # area_fraction

    # Damping Parameters    
    1, # T_surge
    1, # T_sway
    1, # T_heave
    2, # T_yaw
    1, # zeta_roll
    1, # zeta_pitch
    1, # Cd
    1, # e
    1, # r44

    # Fin Parameters
    1, # S_fin
    1, # x_fin
    1, # fin_center
    1, # CL_delta_r
    # 0.8, # CL_delta_s
    0.5, # T_delta

    # Propellor Parameters
    # 0.3, # D_prop
    # 0.2, # t_prop
    # 1, # Ja_max
    # 1, # Va
    # 1, # KT_0
    # 1, # KQ_0
    # 1, # KT_max
    # 1, # KQ_max
    # 1, # nMax
    1, # T_n
    ]

true_params = [
    # Environment Parameters
    # 1000, # rho
    0.5, # V_c
    np.pi/6, # beta_c

    # Physical Parameters
    0, 0, 0.02, # r_bg
    # 0, 0, 0, # r_bb
    # 16, # m
    # 1.6, # L
    0.19, # diam
    0.8, # area_fraction

    # Damping Parameters    
    20, # T_surge
    20, # T_sway
    20, # T_heave
    1, # T_yaw
    0.3, # zeta_roll
    0.8, # zeta_pitch
    0.42, # Cd
    0.6, # e
    0.3, # r44

    # Fin Parameters
    0.00697, # S_fin
    -0.8, # x_fin
    0.07, # fin_center
    0.5, # CL_delta_r
    # 0.7, # CL_delta_s
    0.1, # T_delta

    # Propellor Parameters
    # 0.14, # D_prop
    # 0.1, # t_prop
    # 0.6632, # Ja_max
    # 0.944, # Va
    # 0.4566, # KT_0
    # 0.07, # KQ_0
    # 0.1798, # KT_max
    # 0.0312, # KQ_max
    # 2000, # nMax
    0.1, # T_n
    ]

###### OPTIMIZE ######
t = time.time()
opt_params = opt.least_squares(residuals, params_init, args=(true_states, commands, timestep), method='lm')
# opt_params = opt.minimize(cost, params_init, args=(true_states, commands, timestep), method='Nelder-Mead')
time_elapsed = time.time() - t

print("Final Parameters:")
print(np.round(np.array(opt_params.x), 6))
print("Iterations: ", len(run_history)-1)
print("Time Elapsed: ", time_elapsed)
print("Cost: ", cost(opt_params.x, true_states, commands, timestep))


###### PLOT STATES ######
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

# Set Up Plotting
time = np.arange(0, len(true_states)*timestep/16, timestep)
fig = plt.figure()
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
true_states = true_states.reshape(-1, 16)
positions = [state[:3] for state in true_states]
euler_angles = [state[3:6] for state in true_states]
global_positions = body_to_global(positions, euler_angles)
ax1.plot(time, [pos[0] for pos in global_positions], label='True', color='black', linewidth=2)
ax2.plot(time, [pos[1] for pos in global_positions], label='True', color='black', linewidth=2)
ax3.plot(time, [pos[2] for pos in global_positions], label='True', color='black', linewidth=2)

# Expand history into individual states
for i, run in enumerate(run_history[1:-1]):
    if i%300 == 0:
        run = run.reshape(-1, 16)
        # Extract Positions
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

ax1.legend()
ax2.legend()
ax3.legend()
# plt.tight_layout()
plt.title('State Estimates Over Optimization Iterations')
plt.show()








# Won't converge: m, L, CL_delta_s
# Sensitive to intial conditions: CL_delta_r, e
# Maybe identifiable but takes too long: D_prop, t_prop, 
# Unidentifiable (so far): rho
# not worth identifying: deltaMax_r/s, nMax, all the motor params
