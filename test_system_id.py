import pytest
import numpy as np
import system_id

def test_generate_commands_ouput():
    desired_output = np.array([[1,2,3,4],
                                [1,2,3,4],
                                [1,2,3,4],
                                [1,2,3,4],
                                [5,6,7,8],
                                [5,6,7,8]])
    semantic_commands = [([1,2,3,4], 4), ([5,6,7,8], 2)]
    output = system_id.generate_commands(semantic_commands)
    assert np.array_equal(output, desired_output)
    print("test_generate_commands passed")

def test_generate_commands_bad_input():
    too_long = [([1,2,3,4,5], 4)]
    with pytest.raises(AssertionError):
        system_id.generate_commands(too_long)
    too_short = [([1,2,3], 4)]
    with pytest.raises(AssertionError):
        system_id.generate_commands(too_short)
    not_number = [([1,2,3,'a'], 4)]
    with pytest.raises(AssertionError):
        system_id.generate_commands(not_number)
    not_int = [([1,2,3,4], 4.5)]
    with pytest.raises(AssertionError):
        system_id.generate_commands(not_int)
