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
    assert np.allclose(new_u_actual,command)
    assert np.allclose(new_u_actual_dot,(1/timestep_propeller)*command)

    coug1.u_actual = np.array([0,0,0,0],dtype=float)
    first_u_dot = new_u_actual_dot.copy()

    timesteprudder, timestep_propeller = timesteprudder / 2, timestep_propeller /2
    new_u_actual,new_u_actual_dot = coug1.actuator_dynamics(timesteprudder, command, coug1.u_actual)
    assert np.allclose(new_u_actual, command / 2)
    assert np.allclose(new_u_actual_dot,first_u_dot)

def test_saturate_actuator():
    coug1 = coug.Coug()
    command = [coug1.deltaMax_r]*3
    command.append(coug1.nMax)
    maxed_command = np.array(command)
    over_saturated = maxed_command*2

    coug1.u_actual, _ = coug1.actuator_dynamics(coug1.T_delta, over_saturated, coug1.u_actual)
    coug1.u_actual = coug1.saturate_actuator(coug1.u_actual)

    assert np.all(np.isclose(coug1.u_actual,maxed_command))

def test_state_update():
    coug1 = coug.Coug()
    eta, nu, uactual = coug1.eta.copy(), coug1.nu.copy(), coug1.u_actual.copy()
    state = np.concatenate((eta,nu,uactual))
    new_state = state + .2*np.ones_like(state)
    coug1.stateUpdate(new_state)
    assert np.allclose(eta,coug1.eta - .2*np.ones_like(eta))
    assert np.allclose(nu, coug1.nu - .2*np.ones_like(nu))
    assert np.allclose(uactual, coug1.u_actual - .2*np.ones_like(uactual))

def test_state_update_saturated():
    coug1 = coug.Coug()
    eta, nu, uactual = coug1.eta.copy(), coug1.nu.copy(), coug1.u_actual.copy()
    state = np.concatenate((eta,nu,uactual))
    new_state = state + 30000*np.ones_like(state)
    coug1.stateUpdate(new_state)
    command = [coug1.deltaMax_r]*3
    command.append(coug1.nMax)
    maxed_command = np.array(command)
    assert np.allclose(coug1.u_actual, maxed_command)
