import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

from train import train_2link, load_prev_training, train_compositional_env_base_model, train_subsets_base_model, train_subsets_held_out_base_model
from train import train_cog
from env_inp_out import plot_task_input_output_cog
import config
import tqdm as tqdm

def run_train_compositional_env_base_model():
    # Use input size for original network, will manually change it using add_new_rule_inputs
    hp = {"hid_size": 256, "epochs": 75000}

    load_model_path = "checkpoints/rnn256_softplus_sd1e-3_ma1e-2"
    load_model_file = "rnn256_softplus_sd1e-3_ma1e-2.pth"

    save_model_path = "checkpoints/base_rnn_composable_env_trainloss"
    save_model_file = "base_rnn_composable_env_trainloss.pth"

    print("TRAINING BASE MODEL ON COMPOSITIONAL ENV")
    # leave hp as default
    train_compositional_env_base_model(
        load_model_path, 
        load_model_file, 
        save_model_path, 
        save_model_file,
        hp=hp
    )


def run_train_subsets_base_model():
    """
        This will run training on a subset of the environments with sinusoidinv and figure8inv held out 
        for later transfer learning.
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    hp = {"hid_size": 256}

    model_path = "checkpoints/rnn256_softplus_heldout"
    model_file = "rnn256_softplus_heldout.pth"

    print("TRAINING BASE MODEL ON TASK SUBSETS FOR HELD OUT TESTING")
    # leave hp as default
    train_subsets_base_model(
        model_path, 
        model_file, 
        hp=hp
    )


def run_train_subsets_held_out_base_model():
    """
        This will run training on environments sinusoidinv and figure8inv
        with fixed hidden and input weights except for rule inputs
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    hp = {"hid_size": 256, "epochs": 75000}

    load_model_path = "checkpoints/rnn256_softplus_heldout"
    load_model_file = "rnn256_softplus_heldout.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_transfer"
    save_model_file = "rnn256_softplus_heldout_transfer.pth"

    print("TRAINING BASE MODEL ON TASK SUBSETS WITH TRANSFER")
    # leave hp as default
    train_subsets_held_out_base_model(
        load_model_path, 
        load_model_file, 
        save_model_path, 
        save_model_file, 
        hp=hp
    )

def train_rnn128_softplus():
    hp = {"hid_size": 128}
    model_path = "checkpoints/rnn128_softplus"
    model_file = "rnn128_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 128 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_softplus():
    hp = {"hid_size": 256}
    model_path = "checkpoints/rnn256_softplus_sd1e-3_ma1e-2"
    model_file = "rnn256_softplus_sd1e-3_ma1e-2.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_softplus_noreg():
    hp = {
        "hid_size": 256, 
        "l1_rate": 0, 
        "l1_weight": 0,
        "l1_muscle_act": 0,
        "simple_dynamics_weight": 0
    }
    model_path = "checkpoints/rnn256_softplus_noreg"
    model_file = "rnn256_softplus_noreg.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS NO REG")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_softplus():
    model_path = "checkpoints/rnn512_softplus"
    model_file = "rnn512_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file)

def train_rnn128_relu():
    hp = {"hid_size": 128, "activation_name": "relu"}
    model_path = "checkpoints/rnn128_relu"
    model_file = "rnn128_relu.pth"
    print("TRAINING RNN WITH RELU AND 128 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_relu():
    hp = {"hid_size": 256, "activation_name": "relu"}
    model_path = "checkpoints/rnn256_relu"
    model_file = "rnn256_relu.pth"
    print("TRAINING RNN WITH RELU AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_relu():
    hp = {"activation_name": "relu"}
    model_path = "checkpoints/rnn512_relu"
    model_file = "rnn512_relu.pth"
    print("TRAINING RNN WITH RELU AND 512 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn128_tanh():
    hp = {"hid_size": 128, "activation_name": "tanh"}
    model_path = "checkpoints/rnn128_tanh"
    model_file = "rnn128_tanh.pth"
    print("TRAINING RNN WITH TANH AND 128 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_tanh():
    hp = {"hid_size": 256, "activation_name": "tanh"}
    model_path = "checkpoints/rnn256_tanh"
    model_file = "rnn256_tanh.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn256_tanh_noreg():
    hp = {
        "hid_size": 256, 
        "activation_name": "tanh",
        "l1_rate": 0, 
        "l1_weight": 0,
        "l1_muscle_act": 0,
        "simple_dynamics_weight": 0
    }
    model_path = "checkpoints/rnn256_tanh_noreg"
    model_file = "rnn256_tanh_noreg.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS NO REG")
    # leave hp as default
    train_2link(model_path, model_file, hp=hp)

def train_rnn512_tanh():
    hp = {"activation_name": "tanh"}
    model_path = "checkpoints/rnn512_tanh"
    model_file = "rnn512_tanh.pth"
    print("TRAINING RNN WITH TANH AND 512 UNITS")
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

def continue_training(model_name):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_prev_training(model_path, model_file)

def train_go_task():
    hp = {"hid_size": 256, "inp_size": 99}
    model_path = "checkpoints/go_rnn_relu"
    model_file = "go_rnn_relu.pth"
    print("TRAINING GO TASK WITH RNN")
    train_cog(model_path, model_file, hp)

if __name__ == "__main__":

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    if args.experiment == "train_rnn256_softplus":
        train_rnn256_softplus()
    elif args.experiment == "train_rnn256_softplus_noreg":
        train_rnn256_softplus_noreg()
    elif args.experiment == "train_rnn512_softplus":
        train_rnn512_softplus()
    elif args.experiment == "train_rnn128_softplus":
        train_rnn128_softplus()
    elif args.experiment == "train_rnn256_relu":
        train_rnn256_relu()
    elif args.experiment == "train_rnn512_relu":
        train_rnn512_relu()
    elif args.experiment == "train_rnn128_relu":
        train_rnn128_relu()
    elif args.experiment == "train_rnn256_tanh":
        train_rnn256_tanh()
    elif args.experiment == "train_rnn256_tanh_noreg":
        train_rnn256_tanh_noreg()
    elif args.experiment == "train_rnn512_tanh":
        train_rnn512_tanh()
    elif args.experiment == "train_rnn128_tanh":
        train_rnn128_tanh()
    elif args.experiment == "train_gru256":
        train_gru256()
    elif args.experiment == "train_gru512":
        train_gru512()
    elif args.experiment == "train_gru128":
        train_gru128()
    elif args.experiment == "continue_training":
        continue_training(args.model_name)
    elif args.experiment == "run_train_compositional_env_base_model":
        run_train_compositional_env_base_model()
    elif args.experiment == "run_train_subsets_base_model":
        run_train_subsets_base_model()
    elif args.experiment == "run_train_subsets_held_out_base_model":
        run_train_subsets_held_out_base_model()
    elif args.experiment == "train_go_task":
        train_go_task()
    


    else:
        raise ValueError("Experiment not in this file")