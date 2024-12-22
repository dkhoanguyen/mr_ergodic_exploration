# !/usr/bin/python3

import numpy as np
from gymnasium.spaces import Box
from mr_exploration.dynamics.dynamics_base import DynamicsBase


class DoubleIntegrator(DynamicsBase):
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            exploration_space=exploration_space,
            state_idx=[0, 1],
        )

        self._A = np.array([
            [0., 0., 1.0-0.2, 0.],
            [0., 0., 0., 1.0-0.2],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ])

        self._B = np.array([
            [0., 0.],
            [0., 0.],
            [1.0, 0.],
            [0., 1.0]
        ])

    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics
        '''
        return np.dot(self._A, x) + np.dot(self._B, u)

    def step(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        '''
        Basic euler step
        '''
        new_x = x + self.f(x, u) * dt
        return new_x
