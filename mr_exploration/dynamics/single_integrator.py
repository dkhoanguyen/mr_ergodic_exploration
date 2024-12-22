# !/usr/bin/python3

import numpy as np
from gymnasium.spaces import Box
from mr_exploration.dynamics.dynamics_base import DynamicsBase


class SingleIntegrator(DynamicsBase):

    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box,
                 initial_state: np.ndarray):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            exploration_space=exploration_space,
            state_idx=[0, 1],
            initial_state=initial_state
        )

        self._A = np.array([[0., 0.],
                            [0., 0.]]) 

        self._B = np.array([[1.0, 0.],
                            [0., 1.0]])

    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics
        '''
        return np.dot(self._A, x) + np.dot(self._B, u)

    def step(self, u: np.ndarray, dt: float = 0.1):
        self._state = self._state + self.f(self._state, u) * dt
        return self.state.copy()
