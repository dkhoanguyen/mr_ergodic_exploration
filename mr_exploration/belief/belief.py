#!/usr/bin/python3
import numpy as np
from typing import Tuple, List


class Belief:
    def __init__(self, state: np.ndarray, variance: np.ndarray):
        """
        Initialize the belief.

        Parameters:
            state (np.ndarray): The initial estimated state (mean) as a 1D array.
            variance (np.ndarray): The corresponding variance (for each state dimension) as a 1D array.
        """
        self._state = state
        # variance vector (diagonal of the covariance)
        self._variance = variance

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def variance(self) -> np.ndarray:
        return self._variance

    def step(self, mean: np.ndarray, variance: np.ndarray):
        """
        Update the belief with a new mean and variance.

        Parameters:
            mean (np.ndarray): The new estimated state (mean) as a 1D array.
            variance (np.ndarray): The corresponding variance (for each state dimension) as a 1D array.
        """
        z = mean
        var_meas = variance

        # Compute the precision (inverse variance). If variance is infinite then 1/inf = 0.
        prior_precision = 1 / self._variance
        meas_precision = 1 / var_meas

        # Combined (posterior) precision is the sum of the individual precisions.
        combined_precision = prior_precision + meas_precision

        # The new variance is the inverse of the combined precision.
        new_variance = 1 / combined_precision

        # The new state is a weighted sum of the prior and measurement (weighted by precision).
        new_state = new_variance * \
            ((self.state * prior_precision) + (z * meas_precision))

        # Update the belief.
        self._state = new_state
        self._variance = new_variance
