import torch
import numpy as np
from scipy.interpolate import interp1d


def interpolate_trial(ys, desired_x):
    """
    ys is the time series [timesteps, neurons]
    desired x is the total number of points desired after interpolating
    """
    # range for x is somewhat arb, going with 0-1
    xs = torch.linspace(0, 1, ys.shape[0])
    new_xs = torch.linspace(0, 1, desired_x)

    int_neurons = []
    # Loop through each neuron to get single timeseries
    for n in range(ys.shape[1]):
        t_series = interp1d(xs, ys[:, n])(new_xs)
        int_neurons.append(torch.tensor(t_series))
    new_t_series = torch.stack(int_neurons, dim=1)
    return new_t_series


def random_orthonormal_basis(n, seed=None):
    """
    Generate an n‑dimensional random orthonormal basis.

    Parameters
    ----------
    n : int
        Dimension of the ambient space (must be ≥ 1).
    seed : int, optional
        Random‑seed for reproducibility.

    Returns
    -------
    Q : ndarray, shape (n, n)
        Columns form an orthonormal basis (QᵀQ = I).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))  # random matrix
    Q, _ = np.linalg.qr(A)  # QR factorization ⇒ Q is orthonormal

    # Fix possible sign ambiguity so the first non‑zero entry in each column is positive
    # (optional, just for consistency)
    signs = np.sign(np.diag(Q.T @ A))
    Q *= signs

    return Q
