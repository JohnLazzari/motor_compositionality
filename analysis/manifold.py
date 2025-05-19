from train import train_2link
import motornet as mn
from model import RNNPolicy, GRUPolicy
import torch
import os
from utils import load_hp, create_dir, save_fig, load_pickle, interpolate_trial, random_orthonormal_basis
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
import matplotlib.pyplot as plt
import numpy as np
import config
from analysis.clustering import Analysis
import pickle
from analysis.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
import analysis.plot_utils as plot_utils
#from analysis.dPCA import dPCA
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA
from losses import l1_dist
import scipy
from mRNNTorch.analysis import flow_field
import warnings
warnings.filterwarnings("ignore")




def principal_angles(combinations, combination_labels, mode, num_comps=None, control=True):

    """
        Perform manifold analysis (principle angles and VAF)

        params:
            system: "neural" or "muscle"
            epoch: "delay" or "movement"
    """

    angles_dict = {}
    control_list = []

    if mode == "h":
        num_comps = 12 if num_comps is None else num_comps
        baseline_dim = 256
    elif mode == "muscle_acts":
        num_comps = 3 if num_comps is None else num_comps
        baseline_dim = 6

    if control == True:
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T

    for i, combination in enumerate(combinations):
        
        # ------------------------------------ GET PRINCIPLE ANGLES

        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        # Get principle angles
        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))
        angles_dict[combination_labels[i]] = angles

    if control == True:
        # Get principle angles control
        for i in range(5000):
            rand1 = np.random.randint(2, 5000-1)
            rand2 = np.random.randint(0, rand1-1)
            inner_prod_mat = random_bases[rand1] @ random_bases[rand2].T # Should be m x m
            U, s, Vh = np.linalg.svd(inner_prod_mat)
            angles = np.degrees(np.arccos(s))
            control_list.append(angles)
        control_array = np.stack(control_list, axis=0)

        return angles_dict, control_array
    
    else:

        return angles_dict



def vaf_ratio(combinations, mode, num_comps=None, control=True):

    # Only use two muscle PCs for this task, but use three for the one above

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    if mode == "h":
        num_comps = 12 if num_comps is None else num_comps
        baseline_dim = 256
        percentile = 90
    elif mode == "muscle_acts":
        num_comps = 2 if num_comps is None else num_comps
        baseline_dim = 6
        percentile = 90

    if control == True:
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T

    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        #------------------------------------- PUT TO DEVICES 

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task1_data = torch.tensor(task1_data, dtype=torch.float32).to(device)
        task2_data = torch.tensor(task2_data, dtype=torch.float32).to(device)

        pca1_comps = torch.tensor(pca1_comps, dtype=torch.float32).to(device)
        pca2_comps = torch.tensor(pca2_comps, dtype=torch.float32).to(device)
        """

        #random_bases = torch.tensor(random_bases, dtype=torch.float32).to(device)

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        ratio_task1 = across_task_vaf_task1 / within_task_vaf_task1
        vaf_ratio_list.append(ratio_task1)

        across_task_vaf_task2 = (pca1_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        within_task_vaf_task2 = (pca2_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        ratio_task2 = across_task_vaf_task2 / within_task_vaf_task2
        vaf_ratio_list.append(ratio_task2)

        if control == True:
            # ------------------------------------ CONTROL ACROSS TASK VAFs

            # Get random VAFs
            across_task_vaf = (random_bases @ task1_data.T).var(axis=2).sum(axis=1) / task1_data.var(axis=0).sum()
            vaf_ratio_list_control.append(np.percentile(across_task_vaf, percentile) / within_task_vaf_task1)

            # Get random VAFs
            across_task_vaf = (random_bases @ task2_data.T).var(axis=2).sum(axis=1) / task2_data.var(axis=0).sum()
            vaf_ratio_list_control.append(np.percentile(across_task_vaf, percentile) / within_task_vaf_task2)
    
    if control == True:
        return vaf_ratio_list, vaf_ratio_list_control
    else:
        return vaf_ratio_list