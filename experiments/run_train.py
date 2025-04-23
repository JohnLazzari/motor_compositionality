import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

from train import train_2link
import config
import tqdm as tqdm



def train_rnn512_softplus():
    model_path = "checkpoints/rnn512_softplus"
    model_file = "rnn512_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file)

def train_rnn256_softplus():
    hp = {"hid_size": 256}
    model_path = "checkpoints/rnn256_softplus"
    model_file = "rnn256_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_softplus():
    hp = {"hid_size": 1024}
    model_path = "checkpoints/rnn1024_softplus"
    model_file = "rnn1024_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_relu():
    hp = {"activation_name": "relu"}
    model_path = "checkpoints/rnn512_relu"
    model_file = "rnn512_relu.pth"
    print("TRAINING RNN WITH RELU AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_relu():
    hp = {"hid_size": 256, "activation_name": "relu"}
    model_path = "checkpoints/rnn256_relu"
    model_file = "rnn256_relu.pth"
    print("TRAINING RNN WITH RELU AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_relu():
    hp = {"hid_size": 1024, "activation_name": "relu"}
    model_path = "checkpoints/rnn1024_relu"
    model_file = "rnn1024_relu.pth"
    print("TRAINING RNN WITH RELU AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_tanh():
    hp = {"activation_name": "tanh"}
    model_path = "checkpoints/rnn512_tanh"
    model_file = "rnn512_tanh.pth"
    print("TRAINING RNN WITH TANH AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_tanh():
    hp = {"hid_size": 256, "activation_name": "tanh"}
    model_path = "checkpoints/rnn256_tanh"
    model_file = "rnn256_tanh.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn1024_tanh():
    hp = {"hid_size": 1024, "activation_name": "tanh"}
    model_path = "checkpoints/rnn1024_tanh"
    model_file = "rnn1024_tanh.pth"
    print("TRAINING RNN WITH TANH AND 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru512():
    hp = {"network": "gru"}
    model_path = "checkpoints/gru512"
    model_file = "gru512.pth"
    print("TRAINING GRU WITH 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru256():
    hp = {"hid_size": 256, "network": "gru"}
    model_path = "checkpoints/gru256"
    model_file = "gru256.pth"
    print("TRAINING GRU WITH 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_gru1024():
    hp = {"hid_size": 1024, "network": "gru"}
    model_path = "checkpoints/gru1024"
    model_file = "gru1024.pth"
    print("TRAINING GRU WITH 1024 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)




if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "train_rnn256_softplus":
        train_rnn256_softplus()
    elif args.experiment == "train_rnn512_softplus":
        train_rnn512_softplus()
    elif args.experiment == "train_rnn1024_softplus":
        train_rnn1024_softplus()
    elif args.experiment == "train_rnn256_relu":
        train_rnn256_relu()
    elif args.experiment == "train_rnn512_relu":
        train_rnn512_relu()
    elif args.experiment == "train_rnn1024_relu":
        train_rnn1024_relu()
    elif args.experiment == "train_rnn256_tanh":
        train_rnn256_tanh()
    elif args.experiment == "train_rnn512_tanh":
        train_rnn512_tanh()
    elif args.experiment == "train_rnn1024_tanh":
        train_rnn1024_tanh()
    elif args.experiment == "train_gru256":
        train_gru256()
    elif args.experiment == "train_gru512":
        train_gru512()
    elif args.experiment == "train_gru1024":
        train_gru1024()
    else:
        raise ValueError("Experiment not in this file")