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
