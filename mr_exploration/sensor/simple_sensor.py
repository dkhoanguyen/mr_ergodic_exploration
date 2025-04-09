#!/usr/bin/python3
import numpy as np
from typing import Tuple, List


class SimpleSensor:
    """
    Class to simulate a distance sensor for a robot.
    """

    def __init__(self, range: float = 0.1, noise: float = 0.001):
        self._sensing_range = range
        self._noise = noise

    def step(self,
             sensor_state: np.ndarray,
             ground_truth_state: List[np.ndarray]
             ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate one time step of sensor measurement for multiple targets.
        For each ground truth state in the provided list, the sensor returns a tuple with:
          - A noisy estimate of the true state if the target is within the sensing range.
          - A covariance vector (variance for each dimension) that represents the measurement uncertainty.
        If a target is out of range, the sensor returns an array of NaNs for the measurement
        and an array of infinite variances for the covariance vector.

        Parameters:
            sensor_state (np.ndarray): The current sensor state (e.g., robot position) as an array.
            ground_truth_state (List[np.ndarray]): A list containing the true state of each target as an array.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]:
                A list where each element is a tuple (estimated_state, covariance_vector)
                corresponding to each ground truth state.
        """
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Process each ground truth state.
        for gt in ground_truth_state:
            # Compute Euclidean distance from the sensor to the target.
            distance = np.linalg.norm(gt - sensor_state)
            
            if distance <= self._sensing_range:
                # Sigma increases linearly with distance: at zero distance sigma equals the base noise.
                sigma = self._noise * (1 + distance / self._sensing_range)
                # Generate Gaussian noise for each dimension.
                noise = np.random.randn(*gt.shape) * sigma
                estimated = gt + noise
                covariance_vector = sigma**2 * np.ones(gt.shape)
            else:
                # Out-of-range: return NaNs for measurement and infinite variance for the covariance.
                estimated = np.array([0.5] * gt.shape[0])
                covariance_vector = np.full(gt.shape, 1.0)
                
            results.append((estimated, covariance_vector))
            
        return results