import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

from train import train_2link, load_prev_training, train_orthogonal_networks, train_compositional_env_base_model, train_subsets_base_model, train_subsets_held_out_base_model
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

    save_model_path = "checkpoints/rnn256_softplus_heldout_transfer_trainloss"
    save_model_file = "rnn256_softplus_heldout_transfer_trainloss.pth"

    print("TRAINING BASE MODEL ON TASK SUBSETS WITH TRANSFER")
    # leave hp as default
    train_subsets_held_out_base_model(
        load_model_path, 
        load_model_file, 
        save_model_path, 
        save_model_file, 
        hp=hp
    )


def train_dlyhalfreach():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyhalfreach"
    model_file = "rnn_dlyhalfreach.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON HALFREACH")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyHalfReach", hp=hp)

def train_dlyhalfcircleclk():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyhalfcircleclk"
    model_file = "rnn_dlyhalfcircleclk.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON HALFCIRCLECLK")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyHalfCircleClk", hp=hp)

def train_dlyhalfcirclecclk():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyhalfcirclecclk"
    model_file = "rnn_dlyhalfcirclecclk.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON HALFCIRCLECCLK")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyHalfCircleCClk", hp=hp)

def train_dlysinusoid():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlysinusoid"
    model_file = "rnn_dlysinusoid.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON SINUSOID")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlySinusoid", hp=hp)

def train_dlysinusoidinv():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlysinusoidinv"
    model_file = "rnn_dlysinusoidinv.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON SINUSOIDINV")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlySinusoidInv", hp=hp)

def train_dlyfullreach():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyfullreach"
    model_file = "rnn_dlyfullreach.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON FULLREACH")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyFullReach", hp=hp)

def train_dlyfullcircleclk():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyfullcircleclk"
    model_file = "rnn_dlyfullcircleclk.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON FULLCIRCLECLK")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyFullCircleClk", hp=hp)

def train_dlyfullcirclecclk():
    hp = {"hid_size": 100, "epochs": 50000}
    model_path = "checkpoints/rnn_dlyfullcirclecclk"
    model_file = "rnn_dlyfullcirclecclk.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON FULLCIRCLECCLK")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyFullCircleCClk", hp=hp)

def train_dlyfigure8():
    hp = {"hid_size": 100, "epochs": 75000}
    model_path = "checkpoints/rnn_dlyfigure8"
    model_file = "rnn_dlyfigure8.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON FIGURE8")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyFigure8", hp=hp)

def train_dlyfigure8inv():
    hp = {"hid_size": 100, "epochs": 75000}
    model_path = "checkpoints/rnn_dlyfigure8inv"
    model_file = "rnn_dlyfigure8inv.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 50 UNITS ON FIGURE8INV")
    # leave hp as default
    train_orthogonal_networks(model_path, model_file, "DlyFigure8Inv", hp=hp)

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

def continue_training(model_name):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    load_prev_training(model_path, model_file)



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
    elif args.experiment == "train_rnn256_tanh_noreg":
        train_rnn256_tanh_noreg()
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
    elif args.experiment == "continue_training":
        continue_training(args.model_name)
    

    elif args.experiment == "train_dlyhalfreach":
        train_dlyhalfreach()
    elif args.experiment == "train_dlyhalfcircleclk":
        train_dlyhalfcircleclk()
    elif args.experiment == "train_dlyhalfcirclecclk":
        train_dlyhalfcirclecclk()
    elif args.experiment == "train_dlysinusoid":
        train_dlysinusoid()
    elif args.experiment == "train_dlysinusoidinv":
        train_dlysinusoidinv()
    elif args.experiment == "train_dlyfullreach":
        train_dlyfullreach()
    elif args.experiment == "train_dlyfullcircleclk":
        train_dlyfullcircleclk()
    elif args.experiment == "train_dlyfullcirclecclk":
        train_dlyfullcirclecclk()
    elif args.experiment == "train_dlyfigure8":
        train_dlyfigure8()
    elif args.experiment == "train_dlyfigure8inv":
        train_dlyfigure8inv()

    elif args.experiment == "run_train_compositional_env_base_model":
        run_train_compositional_env_base_model()
    elif args.experiment == "run_train_subsets_base_model":
        run_train_subsets_base_model()
    elif args.experiment == "run_train_subsets_held_out_base_model":
        run_train_subsets_held_out_base_model()

    else:
        raise ValueError("Experiment not in this file")