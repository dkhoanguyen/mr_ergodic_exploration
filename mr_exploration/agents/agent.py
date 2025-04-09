# !/usr/bin/python3

import numpy as np
from mr_exploration.dynamics.dynamics_base import DynamicsBase
from mr_exploration.controllers.ergodic_controller import RTErgodicController
from mr_exploration.fourier_metric.distribution import Distribution


class Agent():
    def __init__(self,
                 initial_state: np.ndarray,
                 dynamics: DynamicsBase,
                 controller: RTErgodicController,
                 agent_idx: int,
                 total_agents: int = 1,
                 max_speed: float = 1.0):
        self._agent_idx = agent_idx
        self._dynamics = dynamics
        self._controller = controller

        self._total_agents = total_agents
        self._max_speed = max_speed

        # Reset and initialized
        self._trajectory = []
        self._ergodic_metrics = []
        self._ck_list = [None] * total_agents

        # self._t_dist: TargetDist = None
        self._dist = Distribution()
        self._t_dist_has_updated = False

        self._state = initial_state

    @property
    def t_dist(self):
        return self._t_dist

    @t_dist.setter
    def t_dist(self, value: Distribution):
        self._t_dist = value

    @property
    def idx(self):
        return self._agent_idx
    
    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def run(self, steps: int = 200, dt: float = 0.1):
        # First calculate the Fourier coefficients for the target distribution
        if self._t_dist:
            self._controller.set_t_dist(self._t_dist)
            # self._t_dist_has_updated = False

        # Check communication status for all agents
        comm_link = all(_ck is not None for _ck in self._ck_list)

        # Update ck in the controller and compute control input
        if comm_link:
            u = self._controller.step(
                self._state.copy(), dt, self._ck_list, self._agent_idx)
        else:
            u = self._controller.step(self._state.copy(), dt)
        
        # Enforce maximum speed constraint during optimization
        u_speed = np.linalg.norm(u)
        if u_speed > self._max_speed:
            u = u / u_speed * self._max_speed

        # Store the Fourier coefficients (ck) for sharing
        current_ck = self._controller.ck.copy()
        self.update_ck(self._agent_idx, current_ck)

        # Update the state using the control input
        self._state = self._dynamics.step(x=self._state.copy(),u=u,dt=dt)
        self._trajectory.append(self._state.copy())

        # # Calculate and store the ergodic metric
        ergodic_metric = np.sum(
            self._controller.lamk * (current_ck - self._controller.phik)**2)
    

    def update_ck(self, agent_idx: int, ck: float):
        self._ck_list[agent_idx] = ck
