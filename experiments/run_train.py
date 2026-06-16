import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")
import pickle

import matplotlib.pyplot as plt
import numpy as np

from modules.multitask_training import MultitaskTrainer
from modules.models import RNNPolicy
from modules.envs.reach import Reach
from modules.envs.clk_curved_reach import ClkCurvedReach
from modules.envs.cclk_curved_reach import CClkCurvedReach
from modules.envs.sinusoid import Sinusoid
from modules.envs.inv_sinusoid import InvSinusoid
from modules.envs.reach_back import ReachBack
from modules.envs.clk_cycle import ClkCycle
from modules.envs.cclk_cycle import CClkCycle
from modules.envs.figure_eight import Figure8
from modules.envs.inv_figure_eight import InvFigure8
import config
import tqdm as tqdm


def run_train_subsets_all_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    mult_train = MultitaskTrainer()
    # Use input size for original network, will manually change it using add_new_rule_inputs
    env_dict = {
        "Reach": Reach,
        "ClkCurvedReach": ClkCurvedReach,
        "Sinusoid": Sinusoid,
        "InvSinusoid": InvSinusoid,
        "ReachBack": ReachBack,
        "ClkCycle": ClkCycle,
        "Figure8": Figure8,
        "InvFigure8": InvFigure8,
    }

    model_path = "checkpoints/rnn256_softplus_heldout"
    model_file = "rnn256_softplus_heldout.pth"

    print("TRAINING BASE MODEL ON ALL TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_nocr_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    mult_train = MultitaskTrainer()
    env_dict = {
        "Reach": Reach,
        "Sinusoid": Sinusoid,
        "InvSinusoid": InvSinusoid,
        "ReachBack": ReachBack,
        "Figure8": Figure8,
        "InvFigure8": InvFigure8,
    }

    model_path = "checkpoints/rnn256_softplus_heldout_nocr"
    model_file = "rnn256_softplus_heldout_nocr.pth"

    print("TRAINING BASE MODEL ON NOCR TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_nosin_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    mult_train = MultitaskTrainer()
    env_dict = {
        "Reach": Reach,
        "ClkCurvedReach": ClkCurvedReach,
        "ReachBack": ReachBack,
        "ClkCycle": ClkCycle,
    }

    model_path = "checkpoints/rnn256_softplus_heldout_nosin"
    model_file = "rnn256_softplus_heldout_nosin.pth"

    print("TRAINING BASE MODEL ON NOSIN TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_reach_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    mult_train = MultitaskTrainer()
    env_dict = {"Reach": Reach, "ReachBack": ReachBack}

    model_path = "checkpoints/rnn256_softplus_heldout_reach"
    model_file = "rnn256_softplus_heldout_reach.pth"

    print("TRAINING BASE MODEL ON REACH TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_cr_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    mult_train = MultitaskTrainer()
    env_dict = {
        "ClkCurvedReach": ClkCurvedReach,
        "ClkCycle": ClkCycle,
    }

    model_path = "checkpoints/rnn256_softplus_heldout_cr"
    model_file = "rnn256_softplus_heldout_cr.pth"

    print("TRAINING BASE MODEL ON CR TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_sin_base_model():
    """
    This will run training on a subset of the environments with sinusoidinv and figure8inv held out
    for later transfer learning.
    """
    # Use input size for original network, will manually change it using add_new_rule_inputs
    mult_train = MultitaskTrainer()
    env_dict = {
        "Sinusoid": Sinusoid,
        "InvSinusoid": InvSinusoid,
        "Figure8": Figure8,
        "InvFigure8": InvFigure8,
    }

    model_path = "checkpoints/rnn256_softplus_heldout_sin"
    model_file = "rnn256_softplus_heldout_sin.pth"

    print("TRAINING BASE MODEL ON SIN TASK SUBSETS FOR HELD OUT TESTING")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def run_train_subsets_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout"
    load_model_file = "rnn256_softplus_heldout.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_transfer"
    save_model_file = "rnn256_softplus_heldout_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING BASE MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def run_train_subsets_nocr_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout_nocr"
    load_model_file = "rnn256_softplus_heldout_nocr.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_nocr_transfer"
    save_model_file = "rnn256_softplus_heldout_nocr_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING NOCR MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def run_train_subsets_nosin_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout_nosin"
    load_model_file = "rnn256_softplus_heldout_nosin.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_nosin_transfer"
    save_model_file = "rnn256_softplus_heldout_nosin_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING NOSIN MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def run_train_subsets_reach_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout_reach"
    load_model_file = "rnn256_softplus_heldout_reach.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_reach_transfer"
    save_model_file = "rnn256_softplus_heldout_reach_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING REACH MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def run_train_subsets_cr_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout_cr"
    load_model_file = "rnn256_softplus_heldout_cr.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_cr_transfer"
    save_model_file = "rnn256_softplus_heldout_cr_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING CR MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def run_train_subsets_sin_held_out_base_model():
    """
    This will run training on environments sinusoidinv and figure8inv
    with fixed hidden and input weights except for rule inputs
    """
    mult_train = MultitaskTrainer()

    load_model_path = "checkpoints/rnn256_softplus_heldout_sin"
    load_model_file = "rnn256_softplus_heldout_sin.pth"

    save_model_path = "checkpoints/rnn256_softplus_heldout_sin_transfer"
    save_model_file = "rnn256_softplus_heldout_sin_transfer.pth"

    env_dict = {
        "CClkCurvedReach": CClkCurvedReach,
        "CClkCycle": CClkCycle,
    }

    print("TRAINING SIN MODEL ON TASK SUBSETS WITH TRANSFER")
    mult_train.train(
        save_model_path,
        save_model_file,
        env_dict=env_dict,
        load_model=True,
        load_model_path=load_model_path,
        load_model_file=load_model_file,
        transfer=True,
    )


def train_rnn128_softplus():
    mult_train = MultitaskTrainer(hid_size=128)
    model_path = "checkpoints/rnn128_softplus"
    model_file = "rnn128_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 128 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_softplus():
    mult_train = MultitaskTrainer()
    model_path = "checkpoints/rnn256_softplus"
    model_file = "rnn256_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_softplus_resevoir():
    mult_train = MultitaskTrainer(resevoir=True, hid_size=256, spectral_radius=1.3)
    env_dict = {
        "Reach": Reach,
        "ClkCurvedReach": ClkCurvedReach,
        "CClkCurvedReach": CClkCurvedReach,
        "Sinusoid": Sinusoid,
        "InvSinusoid": InvSinusoid,
    }
    model_path = "checkpoints/rnn256_softplus_resevoir"
    model_file = "rnn256_softplus_resevoir.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS RESEVOIR")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def train_rnn256_softplus_echo():
    mult_train = MultitaskTrainer(
        resevoir=True, hid_size=256, sparsity=0.99, spectral_radius=1.3
    )
    env_dict = {
        "Reach": Reach,
        "ClkCurvedReach": ClkCurvedReach,
        "CClkCurvedReach": CClkCurvedReach,
        "Sinusoid": Sinusoid,
        "InvSinusoid": InvSinusoid,
    }
    model_path = "checkpoints/rnn256_softplus_echo"
    model_file = "rnn256_softplus_echo.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS ECHO")
    mult_train.train(model_path, model_file, env_dict=env_dict)


def train_rnn256_softplus_nofeedback():
    mult_train = MultitaskTrainer(zero_feedback=True, inp_size=14)
    model_path = "checkpoints/rnn256_softplus_nofeedback"
    model_file = "rnn256_softplus_nofeedback.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS NO FEEDBACK")
    mult_train.train(model_path, model_file)


def train_rnn256_softplus_noreg():
    mult_train = MultitaskTrainer(
        l1_rate=0, l1_weight=0, l1_muscle_act=0, simple_dynamics_weight=0
    )
    model_path = "checkpoints/rnn256_softplus_noreg"
    model_file = "rnn256_softplus_noreg.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 256 UNITS NO REG")
    mult_train.train(model_path, model_file)


def train_rnn512_softplus():
    mult_train = MultitaskTrainer(hid_size=512)
    model_path = "checkpoints/rnn512_softplus"
    model_file = "rnn512_softplus.pth"
    print("TRAINING RNN WITH SOFTPLUS AND 512 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn128_relu():
    mult_train = MultitaskTrainer(hid_size=128, activation_name="relu")
    model_path = "checkpoints/rnn128_relu"
    model_file = "rnn128_relu.pth"
    print("TRAINING RNN WITH RELU AND 128 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_relu():
    mult_train = MultitaskTrainer(activation_name="relu")
    model_path = "checkpoints/rnn256_relu"
    model_file = "rnn256_relu.pth"
    print("TRAINING RNN WITH RELU AND 256 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn512_relu():
    mult_train = MultitaskTrainer(hid_size=512, activation_name="relu")
    model_path = "checkpoints/rnn512_relu"
    model_file = "rnn512_relu.pth"
    print("TRAINING RNN WITH RELU AND 512 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn128_tanh():
    mult_train = MultitaskTrainer(hid_size=128, activation_name="tanh")
    model_path = "checkpoints/rnn128_tanh"
    model_file = "rnn128_tanh.pth"
    print("TRAINING RNN WITH TANH AND 128 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_tanh():
    mult_train = MultitaskTrainer(activation_name="tanh")
    model_path = "checkpoints/rnn256_tanh"
    model_file = "rnn256_tanh.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_tanh_noreg():
    mult_train = MultitaskTrainer(
        activation_name="tanh",
        l1_rate=0,
        l1_weight=0,
        l1_muscle_act=0,
        simple_dynamics_weight=0,
    )
    model_path = "checkpoints/rnn256_tanh_noreg"
    model_file = "rnn256_tanh_noreg.pth"
    print("TRAINING RNN WITH TANH AND 256 UNITS NO REG")
    mult_train.train(model_path, model_file)


def train_rnn512_tanh():
    mult_train = MultitaskTrainer(hid_size=512, activation_name="tanh")
    model_path = "checkpoints/rnn512_tanh"
    model_file = "rnn512_tanh.pth"
    print("TRAINING RNN WITH TANH AND 512 UNITS")
    mult_train.train(model_path, model_file)


def train_gru128():
    mult_train = MultitaskTrainer(network="gru", hid_size=128)
    model_path = "checkpoints/gru128"
    model_file = "gru128.pth"
    print("TRAINING GRU WITH 128 UNITS")
    mult_train.train(model_path, model_file)


def train_gru512():
    mult_train = MultitaskTrainer(network="gru", hid_size=512)
    model_path = "checkpoints/gru512"
    model_file = "gru512.pth"
    print("TRAINING GRU WITH 512 UNITS")
    mult_train.train(model_path, model_file)


def train_gru256():
    mult_train = MultitaskTrainer(network="gru")
    model_path = "checkpoints/gru256"
    model_file = "gru256.pth"
    print("TRAINING GRU WITH 256 UNITS")
    mult_train.train(model_path, model_file)


def train_rnn256_softplus_kinematics():
    mult_train = MultitaskTrainer(
        inp_size=14,
        epochs=100_000,
        l1_rate=1e-4,
        l1_weight=1e-4,
        simple_dynamics_weight=1e-4,
    )
    model_path = "checkpoints/rnn256_softplus_kinematics"
    model_file = "rnn256_softplus_kinematics.pth"
    data_path = "checkpoints/rnn256_softplus/muscle_act_data.pkl"
    print("TRAINING RNN ON SUPERVISED MUSCLE ACTIVITY")
    mult_train.train_kinematics(model_path, model_file, data_path=data_path)


def continue_training(model_name):
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"

    with open(f"{model_name}_mult_train.pkl", "rb") as inp:
        mult_train = pickle.load(inp)

    mult_train.train(model_path, model_file, load_model=True, load_optim=True)


if __name__ == "__main__":
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "train_rnn256_softplus":
        train_rnn256_softplus()
    elif args.experiment == "train_rnn256_softplus_resevoir":
        train_rnn256_softplus_resevoir()
    elif args.experiment == "train_rnn256_softplus_echo":
        train_rnn256_softplus_echo()
    elif args.experiment == "train_rnn256_softplus_noreg":
        train_rnn256_softplus_noreg()
    elif args.experiment == "train_rnn256_softplus_nofeedback":
        train_rnn256_softplus_nofeedback()
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
    elif args.experiment == "train_rnn256_softplus_kinematics":
        train_rnn256_softplus_kinematics()
    elif args.experiment == "continue_training":
        continue_training(args.model_name)

    elif args.experiment == "run_train_subsets_all_base_model":
        run_train_subsets_all_base_model()
    elif args.experiment == "run_train_subsets_nocr_base_model":
        run_train_subsets_nocr_base_model()
    elif args.experiment == "run_train_subsets_nosin_base_model":
        run_train_subsets_nosin_base_model()
    elif args.experiment == "run_train_subsets_reach_base_model":
        run_train_subsets_reach_base_model()
    elif args.experiment == "run_train_subsets_cr_base_model":
        run_train_subsets_cr_base_model()
    elif args.experiment == "run_train_subsets_sin_base_model":
        run_train_subsets_sin_base_model()

    elif args.experiment == "run_train_subsets_held_out_base_model":
        run_train_subsets_held_out_base_model()
    elif args.experiment == "run_train_subsets_nocr_held_out_base_model":
        run_train_subsets_nocr_held_out_base_model()
    elif args.experiment == "run_train_subsets_nosin_held_out_base_model":
        run_train_subsets_nosin_held_out_base_model()
    elif args.experiment == "run_train_subsets_reach_held_out_base_model":
        run_train_subsets_reach_held_out_base_model()
    elif args.experiment == "run_train_subsets_cr_held_out_base_model":
        run_train_subsets_cr_held_out_base_model()
    elif args.experiment == "run_train_subsets_sin_held_out_base_model":
        run_train_subsets_sin_held_out_base_model()

    else:
        raise ValueError("Experiment not in this file")
