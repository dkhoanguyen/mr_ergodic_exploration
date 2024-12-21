import numpy as np
from numpy import pi, cos, sin

from gymnasium.spaces import Box


class Basis(object):
    '''
    Cosine basis functions for decomposing distributions
    '''

    def __init__(self, explr_space: Box, num_basis: int = 5):
        # This entire __init__ fnction is equivalent to step 1 in the tutorial
        # but generalized to support multiple cases instead of just a grid
        # step 1: evaluate the fourier basis function over all the grid cells
        self.ld = explr_space.high - explr_space.low
        n = explr_space.shape[0]
        k = np.meshgrid(*[[i for i in range(num_basis)] for _ in range(n)])

        self.k: np.ndarray = np.c_[k[0].ravel(), k[1].ravel()]
        self.hk: np.ndarray = np.zeros(self.k.shape[0])

        for i, k in enumerate(self.k):
            if np.prod(k) < 1e-5:
                self.hk[i] = 1.
            else:
                numerator = np.prod(
                    self.ld * (2.0 * k * np.pi + np.sin(2.0 * k * np.pi)))
                denominator = 16.0 * np.prod(k) * np.pi**2
                self.hk[i] = numerator/denominator
        self.tot_num_basis = num_basis**n

    def fk(self, x: np.ndarray) -> np.ndarray:
        '''
        Calculate the fourier basis value for a given x
        '''
        assert (x.shape[0] == self.ld.shape[0]
                ), 'input dim does not match explr dim'
        return np.prod(np.cos(np.pi*x/self.ld * self.k), 1)  # /self.hk

    def dfk(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Fourier basis function f_k(x) with respect to x.

        The gradient is defined as:

        ∂f_k(x) / ∂x_i = -π * k_i * sin(π * k_i * x_i / dl_i) / dl_i
                        * ∏_{j ≠ i} cos(π * k_j * x_j / dl_j)

        Args:
            x: The input coordinates (1D array of size n, where n is the dimensionality).

        Returns:
            dx: A 2D array of shape (num_basis, n), where each row corresponds to the gradient
                of a Fourier basis function with respect to each dimension.
        """
        dx = np.zeros((self.tot_num_basis, x.shape[0]))
        dx[:, 0] = -self.k[:, 0]*pi*sin(pi * self.k[:, 0] * x[0]/self.ld[0]) * cos(
            pi * self.k[:, 1]*x[1]/self.ld[1])  # /self.hk
        dx[:, 1] = -self.k[:, 1]*pi*sin(pi * self.k[:, 1] * x[1]/self.ld[1]) * cos(
            pi * self.k[:, 0]*x[0]/self.ld[0])  # /self.hk
        return dx
