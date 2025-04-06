import json
import os
import matplotlib.pyplot as plt

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def create_dir(save_path):
    # Get the directory part of the save path
    directory = os.path.dirname(save_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'r') as f:
        hp = json.load(f)
    return hp

def save_fig(save_path, eps=False):
    # Simple function to save figure while creating dir and closing
    create_dir(save_path)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path, format="eps")
    else:
        plt.savefig(save_path)
    plt.close()