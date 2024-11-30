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

    coug = Coug(params)
    etas = []

    for command in u:
        # nu_dot, u_actual_dot = coug.dynamics(command, timestep) 
        coug.step(command, timestep)
        # append coug.eta, coug.nu, coug.u_actual to some lists or something
        etas.append(coug.eta.copy())
    # TODO: What are we returning here? The state eta probably? Not sure if we need nu or u_actual. 

    # Output: A matrix of the true state at each timestep, flattened into a vector
    etas = np.array(etas).flatten()

    return etas

def residuals(params, truth, u, timestep):
    return truth - simulate(params, u, timestep)

def cost(params, truth, u, timestep):
    # TODO Do we need this? 
    return 



### Set Up Simulation ###
timestep = 1/60. #0.01 # (s). TODO: Tune this? 

# Create list of commands
semantic_commands = [([10,0,0,0],1)] # @Carter generate this part by hand 
commands = generate_commands(semantic_commands)

# Simulate with true parameters to get ground truth
# true_state = simulate(None, commands, timestep)

# Initialize parameters to something
params_init = 1 #TODO: figure out how we want the parameter variable to look/act/work


### Run optimization ###
    # At each iteration, the system will call the resuduals function, which will in turn 
    # call the simulate function. That simulation will initialize a coug with some parameters
    # and simulates the full behavior, then calculates the residuals against the true behavior.
    # The optimizer will calculate the gradient and adjust the parameters accordingly. 

# opt_params = opt.least_squares(residuals, params_init, method='lm')
# this is gonna take absolutely forever to run. 
# When we're just getting started, start with a really short list of commands.
# It won't be long enough ot converge, but it will at least not take a milion years just to test. 
# If I were a good programmer I would make unit tests, instead of writing everythig and testing the whole system at once...

# print(opt_params)
# find some convenent way to compare optimal param to actual params
