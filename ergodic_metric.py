### We define a Gaussian mixture model as the spatial probability density function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

mean1 = np.array([0.35, 0.38])
cov1 = np.array([
    [0.01, 0.004],
    [0.004, 0.01]
])
w1 = 0.5

mean2 = np.array([0.68, 0.25])
cov2 = np.array([
    [0.005, -0.003],
    [-0.003, 0.005]
])
w2 = 0.2

mean3 = np.array([0.56, 0.64])
cov3 = np.array([
    [0.008, 0.0],
    [0.0, 0.004]
])
w3 = 0.3


# Define the Gaussian-mixture density function here
def pdf(x):
    return w1 * mvn.pdf(x, mean1, cov1) + \
           w2 * mvn.pdf(x, mean2, cov2) + \
           w3 * mvn.pdf(x, mean3, cov3)

### We are going to use 10 coefficients per dimension --- so 100 index vectors in total
num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T  # this is the set of all index vectors

# define a 1-by-1 2D search space
L_list = np.array([1.0, 1.0])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 100),
    np.linspace(0, L_list[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = 1.0 / 99
dy = 1.0 / 99  # the resolution of the grids

# Compute the coefficients
coefficients = np.zeros(ks.shape[0])  # number of coefficients matches the number of index vectors
for i, k_vec in enumerate(ks):
    # step 1: evaluate the fourier basis function over all the grid cells
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  # we use NumPy's broadcasting feature to simplify computation
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)  # normalization term
    fk_vals /= hk

    # step 2: evaluate the spatial probabilty density function over all the grid cells
    pdf_vals = pdf(grids)  # this can computed ahead of the time

    # step 3: approximate the integral through the Riemann sum for the coefficient
    phik = np.sum(fk_vals * pdf_vals) * dx * dy 
    coefficients[i] = phik

### We can verify the correctness of the coefficients by reconstructing the probability
### density function through the coefficients

def recon_pdf(_num_k_per_dim):
    _ks_dim1, _ks_dim2 = np.meshgrid(
        np.arange(_num_k_per_dim), np.arange(_num_k_per_dim)
    )
    _ks = np.array([_ks_dim1.ravel(), _ks_dim2.ravel()]).T

    _pdf_recon = np.zeros(grids.shape[0])
    for _i, _k_vec in enumerate(_ks):
        _fk_vals = np.prod(np.cos(np.pi * _k_vec / L_list * grids), axis=1)
        _hk = np.sqrt(np.sum(np.square(_fk_vals)) * dx * dy)
        _fk_vals /= _hk

        _phik = np.sum(_fk_vals * pdf_vals) * dx * dy
        
        _pdf_recon += _phik * _fk_vals 
    
    return _pdf_recon

pdf_gt = pdf(grids)  # ground truth density function

# visualize for comparison
fig, axes = plt.subplots(1, 4, figsize=(18,4), dpi=70, tight_layout=True)

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Original PDF')
ax.contourf(grids_x, grids_y, pdf_gt.reshape(grids_x.shape), cmap='Reds')

ax = axes[1]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Reconstructed PDF\n(5 coefficients)')
pdf_recon_k5 = recon_pdf(5)
ax.contourf(grids_x, grids_y, pdf_recon_k5.reshape(grids_x.shape), cmap='Blues')

ax = axes[2]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Reconstructed PDF\n(10 coefficients)')
pdf_recon_k10 = recon_pdf(10)
ax.contourf(grids_x, grids_y, pdf_recon_k10.reshape(grids_x.shape), cmap='Blues')

ax = axes[3]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Reconstructed PDF\n(20 coefficients)')
pdf_recon_k20 = recon_pdf(20)
ax.contourf(grids_x, grids_y, pdf_recon_k20.reshape(grids_x.shape), cmap='Blues')

plt.show()
plt.close()