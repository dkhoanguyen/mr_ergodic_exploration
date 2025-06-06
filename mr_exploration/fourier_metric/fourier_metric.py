# !/usr/bin/python3

from typing import List
import numpy as np
from mr_exploration.controllers.basis import Basis


class FourierMetric:
    @staticmethod
    def convert_phi2phik(
            basis: Basis,
            phi_val: np.ndarray,
            phi_grid: np.ndarray = None) -> np.ndarray:
        '''
        Converts the distribution to the fourier decompositions
        '''
        if len(phi_val.shape) != 1:
            phi_val = phi_val.ravel()
        if phi_grid is None:
            phi_grid = np.meshgrid(*[np.linspace(0, 1., int(np.sqrt(len(phi_val))))
                                     for _ in range(2)])
            phi_grid = np.c_[phi_grid[0].ravel(), phi_grid[1].ravel()]
        assert phi_grid.shape[0] == phi_val.shape[0], 'samples are not the same'

        return np.sum([basis.fk(x) * v for v, x in zip(phi_val, phi_grid)], axis=0)

    @staticmethod
    def convert_phik2phi(
            basis: Basis,
            phik: np.ndarray,
            phi_grid: List[np.ndarray] = None) -> np.ndarray:
        '''
        Reconstructs phi from the Fourier terms
        '''
        if phi_grid is None:
            phi_grid = np.meshgrid(*[np.linspace(0, 1.)
                                     for _ in range(2)])
            phi_grid = np.c_[phi_grid[0].ravel(), phi_grid[1].ravel()]
        phi_val = np.stack([np.dot(basis.fk(x), phik) for x in phi_grid])
        return phi_val

    @staticmethod
    def convert_traj2ck(basis: Basis, xt: list) -> np.ndarray:
        '''
        This utility function converts a trajectory into its time-averaged
        statistics in the Fourier domain
        '''
        N = len(xt)
        return np.sum([basis.fk(x) for x in xt], axis=0) / N

    @staticmethod
    def convert_ck2dist(basis: Basis, ck: np.ndarray, grid=None) -> np.ndarray:
        '''
        This utility function converts a ck into its time-averaged
        statistics
        '''
        if grid is None:
            grid = np.meshgrid(*[np.linspace(0, 1.)
                                 for _ in range(2)])
            grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        val = np.stack([np.dot(basis.fk(x), ck) for x in grid])
        return val
