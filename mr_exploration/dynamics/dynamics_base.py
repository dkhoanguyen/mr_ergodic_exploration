# !/usr/bin/python3

from abc import ABC, abstractmethod
import numpy as np
from gymnasium.spaces import Box


class DynamicsBase:
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box,
                 state_idx: list):
        self._obs_space = observation_space
        self._action_space = action_space
        self._explr_space = exploration_space
        self._explr_idx = state_idx

        # Default linearized state matrix. Must be overriden  in derived class
        self._A = np.eye(2)
        self._B = np.eye(2)

        self._explr_idx = [0,1]

    @property
    def observation_space(self):
        return self._obs_space
    
    @observation_space.setter
    def observation_space(self, value):
        self._obs_space = value

    @property
    def action_space(self):
        return self._action_space
    
    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @property
    def explr_space(self):
        return self._explr_space
    
    @explr_space.setter
    def explr_space(self, value):
        self._explr_space = value

    @property
    def explr_idx(self):
        return self._explr_idx

    def fdx(self, x: np.ndarray, u: np.ndarray):
        return self._A.copy()

    def fdu(self, x: np.ndarray):
        return self._B.copy()

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics
        '''

    @abstractmethod
    def step(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        '''
        '''
