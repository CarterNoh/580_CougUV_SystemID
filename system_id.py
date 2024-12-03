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

    # TODO: make dictionary out of params.
    # custom to this run: this is where we manually define what variables we're going to run system ID on. 
    # Does doing it this way make it hard for the optimizer to compute a gradient or anything like that?
    if params == None:
        param_dict = None
    else:
        param_dict = {'r44': params[0]}
        # etc

    coug = Coug(param_dict)
    etas = []

    for command in u:
        coug.step(command, timestep)
        etas.append(coug.eta.copy())

    # Output: A matrix of the true state at each timestep, flattened into a vector
    etas = np.array(etas).flatten()

    return etas

def residuals(params, truth, u, timestep):
    '''
    At each iteration of the optimization, the system will call the residuals function, which will 
    in turn call the simulate function. That simulation will initialize a coug with some parameters
    and simulates the full behavior, then calculates the residuals against the true behavior. The 
    optimizer will calculate the gradient and adjust the parameters accordingly.
    '''

    return truth - simulate(params, u, timestep)

# def cost(params, truth, u, timestep):
#     # TODO Do we need this? 
#     return 


### Set Up Simulation ###
timestep = 1/60    # (s). Set to 60 fps, what holoocean commonly does.
step_per_sec = 1/timestep

# Create list of commands
semantic_commands = [([ 0, 0, 0, 1000], int(5*step_per_sec)), # Straight
                     ([ 5, 5,-5, 1000], int(5*step_per_sec)), # Negative Roll
                     ([ 0, 0, 0, 1000], int(5*step_per_sec)), # Straight
                     ([-5,-5, 5, 1000], int(5*step_per_sec)), # Positive Roll
                     ([ 0, 0, 0, 1000], int(5*step_per_sec))] # Straight
commands = generate_commands(semantic_commands)

# Simulate with true parameters to get ground truth
true_state = simulate(None, commands, timestep)

# Initialize parameters to something. TODO: actually do this once we decide what parameters we want. 
params_init = [1] #TODO: figure out how we want the parameter variable to look/act/work 

opt_params = opt.least_squares(residuals, params_init, method='lm', args=(true_state, commands, timestep))
# this is gonna take absolutely forever to run. 
# When we're just getting started, start with a really short list of commands.
# It won't be long enough to converge, but it will at least not take a milion years just to test. 
# If I were a good programmer I would make unit tests, instead of writing everything and testing the whole system at once...

print(opt_params)
# find some convenent way to compare optimal param to actual params
