import numpy as np
import scipy.optimize as opt
from coug import Coug
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
            # 'V_c'       : params[],
            # 'beta_c'    : params[],

            # Physical Parameters
            'r_bg'          : params[0:3],
            # 'r_bb'          : params[3:6],
            # 'm'             : params[3],
            # 'L'             : params[3],
            'diam'          : params[3],
            'area_fraction' : params[4],

            # Damping Parameters: 
            'T_surge'       : params[5],
            'T_sway'        : params[6],
            'T_heave'       : params[7],
            'T_yaw'         : params[8],
            'zeta_roll'     : params[9],
            'zeta_pitch'    : params[10],
            'Cd'            : params[11],
            'e'             : params[12],
            'r44'           : params[13],

            # Fin Parameters
            'S_fin'         : params[14],
            'x_fin'         : params[15],
            'fin_center'    : params[16],
            'CL_delta_r'    : params[17],
            # 'CL_delta_s'    : params[18],
            'T_delta'       : params[18],

            # Motor Parameters
            # 'D_prop'        : params[16],
            # 't_prop'        : params[16],
            # 'Ja_max'        : params[16],
            # 'Va'            : params[16],
            # 'KT_0'          : params[],
            # 'KQ_0'          : params[],
            # 'KT_max'        : params[],
            # 'KQ_max'        : params[],
            'T_n'           : params[19],
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





### Set Up Simulation ###
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
    # 1, # V_c
    # 1, # beta_c: only identifiable if V_c =/= 0?

    # Physical Parameters
    .1, .1, .1, # r_bg
    # 0, 0, .02, # r_bb
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
    0.3, # CL_delta_r
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

t = time.time()
opt_params = opt.least_squares(residuals, params_init, args=(true_states, commands, timestep), method='lm')
# opt_params = opt.minimize(cost, params_init, args=(true_states, commands, timestep), method='Nelder-Mead')
time_elapsed = time.time() - t

print("Final Parameters:")
print(opt_params.x)
print("Iterations: ", opt_params.nfev)
print("Time Elapsed: ", time_elapsed)





# Plot x, y, and z trajectories
# take every 20th entry in the state history
# history_truncated = [run_history[i] for i in range(0, len(run_history), 20)]


# true_states = np.array(run_history[0]).reshape(-1, 16)
# est_states = np.array(run_history[-1]).reshape(-1, 16)
# plt.subplot(3,1,1)







# Won't converge: m, L, CL_delta_s
# Sensitive to intial conditions: CL_delta_r, e, 
# Maybe identifiable but takes forever (sloppy params?): D_prop, t_prop, 
# Unidentifiable (so far): rho

# not worth identifying: deltaMax_r/s, nMax, all the motor params
