import numpy as np
import scipy.optimize as opt
from coug import Coug


######## CougUV State Estimation #########

def generate_commands(u_semantic: list) -> np.ndarray:
    '''
    :param u_semantic: A list of duples, where the first item in the duple is a command and the second item is how long to hold that command.
    

    Four control inputs: Rudder angle, left elevator angle, right elevator angle, and thruster rpm. 
    Goal: generate a Nx4 matrix, where each row is the commands at that timestep. 

    Because we want to simulate at a relatively fast timespan, it would be incredibly tedious to generate
    a matrix of commands by hand. Instead we want something that can take in a "semantic" command sequence 
    that includes the desired commands and the desired time sequence, and stretches that out based on the 
    timestep to generate the command matrix. 

    So the u_semantic input would be a list of duples ( ([20, 5, 5, 10], 4), ([0, 0, 0, 20], 10), ...) where the first
    item in the duple is a command and the second item is how long to hold that command. 
    '''
    commands = []
    for command, duration in u_semantic:
        assert len(command) == 4, "Command must be a list of 4 values"
        assert all(isinstance(x, (int, float)) for x in command), "Command values must be integers or floats"
        assert isinstance(duration, int), "Duration must be an integer"
        for _ in range(duration):
            commands.append(command)
    commands = np.array(commands)

    return commands

def simulate(params, u, timestep):

    if len(params) == 0:
        param_dict = None
    else:
        param_dict = {
            # Environment Parameters
            'rho'       : params[0],
            # 'V_c'       : params[],
            # 'beta_c'    : params[],

            # Physical Parameters


            # Damping Parameters: 
            'T_surge'   : params[1],
            'T_sway'    : params[2],
            'T_heave'   : params[3],
            'T_yaw'     : params[4],
            'zeta_roll' : params[5],
            'zeta_pitch': params[6],
            'Cd'        : params[7],
            # 'e'         : params[],
            'r44'       : params[8],

            # Fin Parameters
            'S_fin'     : params[9],
            'x_fin'     : params[10],
            'fin_center': params[11],

            # Motor Parameters
            'T_n'       : params[12],

            
            }

    coug = Coug(param_dict)
    states = []

    for command in u:
        coug.step(command, timestep)
        state = np.concatenate((coug.eta.copy(), coug.u_actual.copy()), axis=0)
        states.append(state)

    # Output: A matrix of the true state at each timestep, flattened into a vector
    states = np.array(states).flatten()

    # Save the states to something external for plotting or illustrative purposes?

    return states

def residuals(params, truth, u, timestep):
    '''
    At each iteration of the optimization, the system will call the residuals function, which will 
    in turn call the simulate function. That simulation will initialize a coug with some parameters
    and simulates the full behavior, then calculates the residuals against the true behavior. The 
    optimizer will calculate the gradient and adjust the parameters accordingly.
    '''

    return truth - simulate(params, u, timestep)






### Set Up Simulation ###
timestep = 1/60    # (s). Set to 60 fps, what holoocean commonly does.
step_per_sec = 1/timestep

# Create list of commands
semantic_commands = [([ 5, 0, 0, 2000], int(0.4*step_per_sec)),]

commands = generate_commands(semantic_commands)

# Simulate with true parameters to get ground truth
true_state = simulate([], commands, timestep)

# Initialize parameters 
params_init = [
    # Environment Parameters
    800, # rho
    # 1, # V_c
    # 1, # beta_c: only identifiable if V_c =/= 0?

    # Physical Parameters

    # Damping Parameters    
    1, # T_surge
    1, # T_sway
    1, # T_heave
    2, # T_yaw
    1, # zeta_roll
    1, # zeta_pitch
    1, # Cd
#    1, # e
    1, # r44

    # Fin Parameters
    1, # S_fin
    1, # x_fin
    1, # fin_center

    # Propellor Parameters
    1, # T_n

    ]

opt_params = opt.least_squares(residuals, params_init, method='lm', args=(true_state, commands, timestep))

print('\n')
print("Final Parameters:")
print(opt_params.x)
print(" Iterations: ", opt_params.nfev)
# Find some better way to present parameters?

### Commands for various parameters
# Straight: r44, T_n, Cd,zeta_roll, zeta_pitch, T_surge, 
# Any ONE fin: S_fin, x_fin, fin_center

# Identifiable but takes forever (add back later): CL_delta_r, CL_delta_s, 
# Unidentifiable (so far): T_delta, e
