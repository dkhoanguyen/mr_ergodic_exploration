import numpy as np
from gymnasium.spaces import Box

from dynamics import Dynamics


class DoubleIntegrator(Dynamics):
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box,
                 initial_state: np.ndarray):
        self._A = np.array([
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ])  # - np.diag([0,0,1,1]) * 0.25

        self._B = np.array([
            [0., 0.],
            [0., 0.],
            [1.0, 0.],
            [0., 1.0]
        ])

        self._obs_space = observation_space
        self._action_space = action_space
        self._explr_space = exploration_space

        self._state: np.ndarray = initial_state

        # The meaning of this variable is that we only care about position
        # in the state vector
        self._explr_indx = [0,1]

    def fdx(self, x: np.ndarray, u: np.ndarray):
        '''
        State linearization
        '''
        return self._A.copy()

    def fdu(self, x: np.ndarray):
        '''
        Control linearization
        '''
        return self._B.copy()

    def reset(self, state: np.ndarray):
        if state is None:
            self._state = np.zeros(self._obs_space.shape[0])
            self._state[:2] = np.random.uniform(0., 0.9, size=(2,))
        else:
            self._state = state
        return self._state.copy

    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics
        '''
        return np.dot(self._A, x) + np.dot(self._B, u)

    def step(self, u: np.ndarray, dt: float):
        '''
        Euler discretization of the linearized dynamic
        '''
        self._state = self._state + self.f(self._state, u) * dt
        return self._state.copy()


if __name__ == "__main__":
    # 4D observation space
    observation_space = Box(np.array([0., 0., -np.inf, -np.inf]),
                            np.array([1.0, 1.0, np.inf, np.inf]),
                            dtype=np.float64)
    # 2D action space - likely the bound of the control
    action_space = Box(np.array([-1., -1.]),
                       np.array([1.0, 1.0]),
                       dtype=np.float64)
    # 2D exploration space
    explr_space = Box(np.array([0., 0.]),
                      np.array([1.0, 1.0]),
                      dtype=np.float64)

    dyn = DoubleIntegrator(observation_space=observation_space,
                           action_space=action_space,
                           exploration_space=explr_space,
                           initial_state=np.array([0, 0, 0, 0]))

    print(dyn.action_space)
