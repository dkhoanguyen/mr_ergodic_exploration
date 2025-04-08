# !/usr/bin/python3

import numpy as np
from gymnasium.spaces import Box
from mr_exploration.dynamics.dynamics_base import DynamicsBase


class DublinCarModel(DynamicsBase):
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box,
                 max_velocity: float = 5.0,
                 max_steering_angle: float = np.pi / 4):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            exploration_space=exploration_space,
            state_idx=[0, 1],
        )

        # State: [x, y, theta, v] (x, y position, orientation, velocity)
        # Action: [a, delta] (acceleration, steering angle)
        self._A = np.eye(4)  # Placeholder for linearized A matrix
        self._B = np.zeros((4, 2))  # Placeholder for linearized B matrix

        self._max_velocity = max_velocity
        self._max_steering_angle = max_steering_angle

    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics for a Dubin's car model
        x: [x, y, theta, v] (state)
        u: [a, delta] (action)
        '''
        a, delta = np.clip(u, [-np.inf, -self._max_steering_angle], [np.inf, self._max_steering_angle])
        theta = x[2]
        v = np.clip(x[3], -self._max_velocity, self._max_velocity)

        dx = np.array([
            v * np.cos(theta),  # x_dot
            v * np.sin(theta),  # y_dot
            v * np.tan(delta),  # theta_dot
            a                  # v_dot
        ])
        return dx

    def linearize(self, x: np.ndarray, u: np.ndarray):
        '''
        Linearizes the dynamics around the state x and input u
        '''
        a, delta = np.clip(u, [-np.inf, -self._max_steering_angle], [np.inf, self._max_steering_angle])
        theta = x[2]
        v = np.clip(x[3], -self._max_velocity, self._max_velocity)

        A = np.array([
            [0, 0, -v * np.sin(theta), np.cos(theta)],
            [0, 0,  v * np.cos(theta), np.sin(theta)],
            [0, 0,  0,                  np.tan(delta)],
            [0, 0,  0,                  0]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [0, v / np.cos(delta)**2],
            [1, 0]
        ])

        self._A = A
        self._B = B

        return A, B

    def step(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        '''
        Basic Euler integration for the Dubin's car model with velocity and steering constraints
        '''
        new_x = x + self.f(x, u) * dt

        # Enforce velocity and steering constraints
        new_x[3] = np.clip(new_x[3], -self._max_velocity, self._max_velocity)  # Velocity constraint
        new_x[2] = np.clip(new_x[2], -self._max_steering_angle, self._max_steering_angle)  # Steering constraint

        return new_x
