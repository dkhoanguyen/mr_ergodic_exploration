# !/usr/bin/python3

import numpy as np
from mr_exploration.dynamics.dynamics_base import DynamicsBase
from mr_exploration.controllers.ergodic_controller import RTErgodicController
from mr_exploration.util.target_dist import TargetDist
from mr_exploration.util.utils import *


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

        self._t_dist: TargetDist = None
        self._t_dist_has_updated = False

        self._state = initial_state

    @property
    def t_dist(self):
        return self._t_dist

    @t_dist.setter
    def t_dist(self, value: TargetDist):
        self._t_dist = value
        self._t_dist_has_updated = True

    @property
    def idx(self):
        return self._agent_idx

    def run(self, steps: int = 200, dt: float = 0.1):
        for step in range(steps):
            # First calculate the Fourier coefficients for the target distribution
            if self._t_dist and self._t_dist_has_updated:
                self._controller.phik = convert_phi2phik(
                    self._controller.basis, self._t_dist.grid_vals, self._t_dist.grid
                )
                self._t_dist_has_updated = False

            # Check communication status for all agents
            comm_link = all(_ck is not None for _ck in self._ck_list)

            # print(f"state b4 step: {self._dynamics.state}")
            # current_state = self._dynamics.state.copy()
            current_state = self._state

            # Update ck in the controller and compute control input
            if comm_link:
                u = self._controller.step(
                    current_state.copy(), dt, self._ck_list, self._agent_idx)
            else:
                u = self._controller.step(current_state.copy(), dt)
            
            self._dynamics._state = current_state.copy()
            # Enforce maximum speed constraint during optimization
            u_speed = np.linalg.norm(u)
            if u_speed > self._max_speed:
                u = u / u_speed * self._max_speed
            # print(u)

            # Store the Fourier coefficients (ck) for sharing
            current_ck = self._controller.ck.copy()
            self._update_ck(self._agent_idx, current_ck)

            # Update the state using the control input
            print(f"state b4 step: {self._state}")
            self._state = self._dynamics.step(x=current_state,u=u,dt=dt)
            print(f"Updated x: {self._state}")
            self._trajectory.append(self._state.copy())

            # Calculate and store the ergodic metric
            ergodic_metric = np.sum(
                self._controller.lamk * (current_ck - self._controller.phik)**2)
            print()

    def _update_ck(self, agent_idx: int, ck: float):
        self._ck_list[agent_idx] = ck
