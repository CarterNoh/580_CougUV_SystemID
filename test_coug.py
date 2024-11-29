import pytest
import numpy as np
import coug

def test_override_params_good():
    coug1 = coug.Coug()

    params = {
        "nu" : np.ones(6, dtype=float),
        "eta" : np.ones(6, dtype=float),
        "u_actual" : np.ones(4, dtype=float),
        "rho" : 0,
        "g" : 0,
        "V_c" : 0,
        "beta_c" : 0,
        "r_bg" : np.ones(3, dtype=float),
        "r_bb" : np.ones(3, dtype=float),
        "m" : 0,
        "L" : 0,
    }

    coug1.override_params(params)
    for key in params.keys():
        if isinstance(params[key], np.ndarray):
            assert np.array_equal(coug1.__dict__[key], params[key])
        else:
            assert coug1.__dict__[key] == params[key]

def test_override_params_bad():
    coug1 = coug.Coug()

    params = {
        "dummyParam": 0
    }
    with pytest.raises(ValueError):
        coug1.override_params(params)
    
def test_actuator_dynamics():
    coug1 = coug.Coug()

    command = np.array([1,1,1,1])
    timesteprudder = coug1.T_delta
    timestep_propeller = coug1.T_n

    assert timesteprudder == timestep_propeller

    new_u_actual, new_u_actual_dot = coug1.actuator_dynamics(timesteprudder, command, coug1.u_actual)
    assert np.all(np.isclose(new_u_actual,command))
    assert np.all(np.isclose(new_u_actual_dot,(1/timestep_propeller)*command))

    coug1.u_actual = np.array([0,0,0,0],dtype=float)
    first_u_dot = new_u_actual_dot.copy()

    timesteprudder, timestep_propeller = timesteprudder / 2, timestep_propeller /2
    new_u_actual,new_u_actual_dot = coug1.actuator_dynamics(timesteprudder, command, coug1.u_actual)
    assert np.all(np.isclose(new_u_actual, command / 2))
    assert np.all(np.isclose(new_u_actual_dot,first_u_dot))

