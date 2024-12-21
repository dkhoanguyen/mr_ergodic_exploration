from abc import ABC, abstractmethod
import numpy as np


class Dynamics(ABC):
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def exploration_space(self):
        return self._explr_space

    @property
    def explr_idx(self):
        return self._explr_indx

    @abstractmethod
    def fdx(self, x: np.ndarray, u: np.ndarray):
        '''
        State linearization
        '''

    @abstractmethod
    def fdu(self, x: np.ndarray):
        '''
        Control linearization
        '''

    @abstractmethod
    def reset(self, state: np.ndarray):
        '''
        '''

    @abstractmethod
    def reset(self, x: np.ndarray):
        '''
        '''

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        '''

    @abstractmethod
    def step(self, u: np.ndarray, dt: float):
        '''
        '''
