import numpy as np
import scipy.optimize as opt
from coug import Coug
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
            'rho'           : params[0],
            'V_c'           : params[1],
            'beta_c'        : params[2],

            # Physical Parameters
            'r_bg'          : params[3:6],
            'r_bb'          : params[6:9],
            # 'm'             : params[],
            # 'L'             : params[],
            'diam'          : params[9],
            'area_fraction' : params[10],

            # Damping Parameters: 
            'T_surge'       : params[11],
            'T_sway'        : params[12],
            'T_heave'       : params[13],
            'T_yaw'         : params[14],
            'zeta_roll'     : params[15],
            'zeta_pitch'    : params[16],
            'Cd'            : params[17],
            'e'             : params[18],
            'r44'           : params[19],

            # Fin Parameters
            'S_fin'         : params[20],
            'x_fin'         : params[21],
            'fin_center'    : params[22],
            'CL_delta'      : params[23],
            'T_delta'       : params[24],

            # Motor Parameters
            # 'D_prop'        : params[],
            't_prop'        : params[25],
            # 'Ja_max'        : params[],
            # 'Va'            : params[],
            # 'KT_0'          : params[],
            # 'KQ_0'          : params[],
            # 'KT_max'        : params[],
            # 'KQ_max'        : params[],
            'T_n'           : params[26],
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
    800, # rho
    1, # V_c
    np.pi*3/4, # beta_c: only identifiable if V_c =/= 0?

    # Physical Parameters
    .1, .1, .1, # r_bg
    .05, .05, .05, # r_bb
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
    1, # CL_delta
    0.5, # T_delta

    # Propellor Parameters
    # 0.3, # D_prop
    0.2, # t_prop
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
    1000, # rho
    0.5, # V_c
    np.pi/6, # beta_c

    # Physical Parameters
    0, 0, 0.02, # r_bg
    0, 0, 0, # r_bb
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
    0.6, # CL_delta
    0.1, # T_delta

    # Propellor Parameters
    # 0.14, # D_prop
    0.1, # t_prop
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

# Save state history to file
np.save('run_history_4.npy', run_history)



########## Notes ##########
## Easily Measured/Specced: 13
# m, L, diam, D_prop, Ja_max, Va, KT_0, KQ_0, KT_max, KQ_max, nMax, deltaMax
## Approximate/Assumed Parameters: 12
# rho, r_bg, r_bb, area_fraction, S_fin, x_fin, fin_center, t_prop
## Unknown Parameters: 15
# V_c, beta_c, T_surge, T_sway, T_heave, T_yaw, zeta_roll, zeta_pitch, Cd, e, r44, CL_delta_r, CL_delta_s, T_delta, T_n
## Of these, we are estimating 22: 1 measurable, all approximate, all unknown


# Won't converge: m, L
# Sensitive to intial conditions: CL_delta, e, r_bb
# Takes a long time: r_bb, t_prop