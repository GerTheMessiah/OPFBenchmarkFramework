""" Integration tests of all the default environments. """

import numpy as np

from mlopf.envs.thesis_envs import SimpleOpfEnv, QMarketEnv, EcoDispatchEnv


def test_simple_opf_integration():
    dummy_env = SimpleOpfEnv()
    dummy_env.reset()
    for _ in range(3):
        act = dummy_env.action_space.sample()
        obs, reward, done, done, info = dummy_env.step(act)
        dummy_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert done
    assert isinstance(info, dict)


def test_qmarket_integration():
    dummy_env = QMarketEnv(**{"eval": True, })
    obs, info = dummy_env.reset()

    for _ in range(6719):
        act = dummy_env.action_space.sample()
        obs_new, reward, done, done, info = dummy_env.step(act)
        obs, info = dummy_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert done
    assert isinstance(info, dict)


def test_eco_dispatch_integration():
    dummy_env = EcoDispatchEnv(eval=True)
    obs, info = dummy_env.reset()

    for _ in range(3):
        act = dummy_env.action_space.sample()
        print(act.shape)
        obs, reward, done, done, info = dummy_env.step(act)
        dummy_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert done
    assert isinstance(info, dict)
