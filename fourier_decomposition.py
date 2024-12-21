import numpy as np

from basis import Basis


def convert_phi2phik(basis: Basis, phi_val: np.ndarray, phi_grid: np.ndarray = None):
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
