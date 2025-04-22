# !/usr/bin/python3

from typing import List
import numpy as np


class Distribution:
    def __init__(self, num_pts: int = 20):
        self._num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self._grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        self._means = [np.array([0.5, 0.5])]
        self._vars = [np.array([0.05, 0.05])]

        self._has_updated = False
        self._grid_values: np.ndarray = self._evaluate(self._grid)

    @property
    def means(self) -> List[np.ndarray]:
        return self._means

    @property
    def vars(self) -> List[np.ndarray]:
        return self._vars
    
    @property
    def grid_vals(self):
        return self._grid_values
    
    @property
    def grid(self):
        return self._grid

    def update(self, means: List[np.ndarray], vars: List[np.ndarray]):
        assert len(means) == len(
            vars), 'Means and vars need to be the same size'
        self._means = means
        self._vars = vars
        self._has_updated = True
        self._grid_values: np.ndarray = self._evaluate(self._grid)

    def get_grid_spec(self):
        xy = []
        for g in self._grid.T:
            xy.append(
                np.reshape(g, newshape=(self._num_pts, self._num_pts))
            )
        return xy, self._grid_values.reshape(self._num_pts, self._num_pts)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])
        for m, v in zip(self._means, self._vars):
            innerds = np.sum((x-m)**2 / v, 1)
            val += np.exp(-innerds/2.0)  # / np.sqrt((2*np.pi)**2 * np.prod(v))
        # normalizes the distribution
        val /= np.sum(val)
        # val -= np.max(val)
        # val = np.abs(val)
        return val
    
    def set_uniform(self):
        """
        Set the grid values to represent a uniform distribution.
        This assigns an equal probability to each point in the grid.
        """
        n = self._grid.shape[0]
        self._grid_values = np.full(n, 1.0/n)
        # print(self._grid_values)
        self._has_updated = True

    def get_subsection(self, center: np.ndarray, radius: float) -> float:
        """
        Calculate the subsection of the grid given an (x,y) position and a radius.
        
        Parameters:
            center (np.ndarray): A 1D numpy array [x, y] defining the center of the subsection.
            radius (float): The radius defining the circular region around the center.
        
        Returns:
            sub_grid (np.ndarray): An array of grid points (shape (M,2)) that lie within the specified radius.
            sub_values (np.ndarray): The corresponding grid values for those grid points (shape (M,)).
        """
        # Compute the Euclidean distance from the center to each grid point.
        distances = np.linalg.norm(self._grid - center, axis=1)
        # Select grid points within the radius.
        mask = distances <= radius
        sub_grid = self._grid[mask]
        sub_values = self._grid_values[mask]
        return sub_grid, sub_values
    