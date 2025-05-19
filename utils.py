import json
import os
import matplotlib.pyplot as plt
import pickle
import scipy
from scipy.interpolate import interp1d
import numpy as np
import torch
from matplotlib import rcParams

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def create_dir(save_path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'r') as f:
        hp = json.load(f)
    return hp

def save_fig(save_path, eps=False): 
    # Tell matplotlib to embed fonts as text, not outlines
    rcParams['pdf.fonttype'] = 42  # 42 = TrueType (editable in Illustrator)
    rcParams['ps.fonttype'] = 42
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path + ".pdf", format="pdf")
    else:
        plt.savefig(save_path + ".png")
    plt.close()

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data

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
    A = rng.standard_normal((n, n))       # random matrix
    Q, _ = np.linalg.qr(A)                # QR factorization ⇒ Q is orthonormal

    # Fix possible sign ambiguity so the first non‑zero entry in each column is positive
    # (optional, just for consistency)
    signs = np.sign(np.diag(Q.T @ A))
    Q *= signs

    return Q
