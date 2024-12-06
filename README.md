# CougUV_SystemID

This project performs system identification to determine the dynamics parameters for the CougUV, a torpedo-type underwater vehicle developed by the BYU FRoSt Lab. 

## Files
- coug.py: Implementation of vehicle dynamics. Based on Thor Fossen's model in "Marine Craft Hydrodynamics and Control". 
- system_id.py: Performs system identification to estimate dynamics parameters. 
- pose_plotter.py: Plots poses over iterations to visualize the optimization process. 
- helper_functions.py: functions for common computations in the dynamics.
- test_coug.py: Unit tests for the dynamics.
- test_system_id.py: unit tests for the system ID process. 
- run_history_27.npy: history of states across iterations for the run estimating 27 parameters.
- results_27.txt: numerical results for run with 27 parameters