from gym.spaces import Box, Dict
import numpy as np
import gym

class BridgeKitchenDummyEnv(gym.Env):
    def __init__(self, from_states=False, add_states=False, num_tasks=1):
        super().__init__()
        obs_dict = dict()
        if not from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.spec = None
        self.action_space = Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)

    def seed(self, seed):
        pass
    
    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}