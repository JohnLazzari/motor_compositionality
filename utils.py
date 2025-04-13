import json
import os
import matplotlib.pyplot as plt
import pickle

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
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path, format="eps")
    else:
        plt.savefig(save_path)
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