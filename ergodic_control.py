import random

import numpy as np
from dynamics import Dynamics
from basis import Basis

class RTErgodicController():
    def __init__(self, dynamics: Dynamics,
                 weights: np.ndarray,
                 horizon: int = 100,
                 num_basis: int = 5,
                 capacity: int = 100000,
                 batch_size: int = 20):
        self.dynamics = dynamics
        self.horizon = horizon
        self.weights = weights
        self.batch_size = batch_size
        self.capacity = capacity

        self.basis = Basis(
            self.dynamics.exploration_space, num_basis=num_basis)
        self.lamk = np.exp(-0.8 * np.linalg.norm(self.basis.k, axis=1))

        
        self.R_inv = np.linalg.inv(weights)

        # Fourier coefficients for spatial distribution
        self._phik = None
        # Fourier coefficients for time-averaged trajectory
        self._ck = None

        self.reset()
        

    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik

    def reset(self):
        self.u_seq = [np.zeros(self.dynamics.action_space.shape) for _ in range(self.horizon)]
        self.replay_buffer = []
        self.replay_position = 0

    def step(self, state: np.ndarray, dt: float):
        # Shift control sequence
        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1] = np.zeros(self.dynamics.action_space.shape)

        # Predict trajectories and collect gradients
        x = self.dynamics.reset(state)
        pred_traj = []
        dfk = []
        fdx = []
        fdu = []

        for t in range(self.horizon):
            pred_traj.append(x[self.dynamics.explr_idx])
            dfk.append(self.basis.dfk(x[self.dynamics.explr_idx]))
            fdx.append(self.dynamics.fdx(x, self.u_seq[t]))
            fdu.append(self.dynamics.fdu(x))
            x = self.dynamics.step(self.u_seq[t])

        # Calculate Fourier coefficients from trajectories
        pred_traj += self._sample_from_replay_buffer(self.batch_size)
        ck = np.mean([self.basis.fk(xt) for xt in pred_traj], axis=0)

        # Fourier coefficient difference
        coeff_diff = self.lamk * (ck - self._phik).reshape(-1, 1)

        # Backward pass for control computation
        rho = np.zeros(self.dynamics.observation_space.shape)
        for t in reversed(range(self.horizon)):
            grad_cost = np.zeros(self.dynamics.observation_space.shape)
            grad_cost[self.dynamics.explr_idx] = np.sum(dfk[t] * coeff_diff, axis=0)
            rho -= dt * (-grad_cost - np.dot(fdx[t].T, rho))
            self.u_seq[t] = -np.dot(np.dot(np.eye(self.dynamics.action_space.shape[0]), fdu[t].T), rho)

            # Control normalization for constraints
            if np.linalg.norm(self.u_seq[t]) > 1.0:
                self.u_seq[t] /= np.linalg.norm(self.u_seq[t])

        # Store current state in the replay buffer
        self._push_to_replay_buffer(state[self.dynamics.explr_idx])
        return self.u_seq[0].copy()


    def _push_to_replay_buffer(self, state):
        """Store a state in the buffer, replacing the oldest if full."""
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        self.replay_buffer[self.replay_position] = state
        self.replay_position = (self.replay_position + 1) % self.capacity

    def _sample_from_replay_buffer(self, batch_size):
        """Sample a random batch from the buffer."""
        if batch_size == -1 or batch_size > len(self.replay_buffer):
            return self.replay_buffer[:]
        return random.sample(self.replay_buffer, batch_size)