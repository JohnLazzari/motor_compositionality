import os
import sys
import torch
import motornet as mn
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import RNNPolicy, GRUPolicy
from losses import l1_dist, l1_rate, l1_weight, l1_muscle_act, simple_dynamics
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
from utils import save_hp, create_dir

DEF_HP = {
    "network": "rnn",
    "inp_size": 28,
    "hid_size": 512,
    "activation_name": "softplus",
    "noise_level_act": 0.1,
    "noise_level_inp": 0.01,
    "constrained": False,
    "dt": 10,
    "t_const": 20,
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 50_000,
    "save_iter": 500,
    "l1_rate": 0.001,
    "l1_weight": 0.001,
    "l1_muscle_act": 0.001,
    "simple_dynamics_weight": 0.001
}

def do_eval(policy, hp):

    env_dict = {
        "DlyHalfReach": DlyHalfReach, 
        "DlyHalfCircleClk": DlyHalfCircleClk, 
        "DlyHalfCircleCClk": DlyHalfCircleCClk, 
        "DlySinusoid": DlySinusoid, 
        "DlySinusoidInv": DlySinusoidInv,
        "DlyFullReach": DlyFullReach,
        "DlyFullCircleClk": DlyFullCircleClk,
        "DlyFullCircleCClk": DlyFullCircleCClk,
        "DlyFigure8": DlyFigure8,
        "DlyFigure8Inv": DlyFigure8Inv
    }

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    # Currently 10 speed conds during testing, 3 for training
    speed_conds = list(np.arange(0, 10))

    total_test_loss = 0
    condition_losses = {}
    for env in env_dict:
        condition_loss = 0
        for speed in speed_conds:

            # initialize batch
            # use 32 to get every possible direction
            x = torch.zeros(size=(32, hp["hid_size"]))
            h = torch.zeros(size=(32, hp["hid_size"]))

            cur_env = env_dict[env](effector=effector)

            # Get first timestep
            obs, info = cur_env.reset(testing=True, options={"batch_size": 32, "reach_conds": np.arange(0, 32), "speed_cond": speed})
            terminated = False

            # initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]

            timestep = 0
            # simulate whole episode
            while not terminated:  # will run until `max_ep_duration` is reached

                with torch.no_grad():
                    x, h, action = policy(obs, x, h)
                    obs, reward, terminated, info = cur_env.step(timestep, action=action)

                xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
                tg.append(info["goal"][:, None, :])  # targets

                timestep += 1

            # concatenate into a (batch_size, n_timesteps, xy) tensor
            xy = torch.cat(xy, axis=1)
            tg = torch.cat(tg, axis=1)

            # Implement loss function
            loss = l1_dist(xy, tg)  # L1 loss on position
            condition_loss += loss.item()
        condition_loss /= len(speed_conds)
        condition_losses[env] = condition_loss
        total_test_loss += condition_loss
    total_test_loss /= len(env_dict)

    print("\n")
    print("Eval Results:")
    for env in condition_losses:
        print(f"Total Loss for Environment {env}| {condition_losses[env]}")
    print(f"Total Testing Loss: {total_test_loss}")
    print("\n")
    
    return total_test_loss


def train_2link(model_path, model_file, hp=None):

    # create model path for saving model and hp
    create_dir(model_path)

    def_hp = DEF_HP
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    # save hyperparameters
    save_hp(hp, model_path)

    device = torch.device("cpu")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    if hp["network"] == "rnn":
        policy = RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            effector.n_muscles, 
            activation_name=hp["activation_name"],
            noise_level_act=hp["noise_level_act"], 
            noise_level_inp=hp["noise_level_inp"], 
            constrained=hp["constrained"], 
            dt=hp["dt"],
            t_const=hp["t_const"],
            device=device
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    losses = []
    interval = 100
    best_test_loss = np.inf

    env_list = [
        DlyHalfReach, 
        DlyHalfCircleClk, 
        DlyHalfCircleCClk, 
        DlySinusoid, 
        DlySinusoidInv,
        DlyFullReach,
        DlyFullCircleClk,
        DlyFullCircleCClk,
        DlyFigure8,
        DlyFigure8Inv
    ]

    probs = [1/len(env_list)] * len(env_list)

    for batch in range(hp["epochs"]):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        rand_env = random.choices(env_list, probs)
        env = rand_env[0](effector=effector)

        # Get first timestep
        obs, info = env.reset(options={"batch_size": hp["batch_size"]})
        terminated = False

        # initial positions and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hs = [h.unsqueeze(1)]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            x, h, action = policy(obs, x, h)
            obs, reward, terminated, info = env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hs.append(h.unsqueeze(1))

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        muscle_acts = torch.cat(muscle_acts, axis=1)
        hs = torch.cat(hs, axis=1)

        # Implement loss function
        loss = l1_dist(xy, tg)  # L1 loss on position
        loss += l1_rate(hs, hp["l1_rate"])
        loss += l1_weight(policy, hp["l1_weight"])
        loss += l1_muscle_act(muscle_acts, hp["l1_muscle_act"])
        loss += simple_dynamics(hs, policy.mrnn, weight=hp["simple_dynamics_weight"])
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], sum(losses[-interval:])/interval))

        if (batch % hp["save_iter"] == 0):
            # Get test loss
            test_loss = do_eval(policy, hp)
            # If current test loss is better than previous, save model and update best loss
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'agent_state_dict': policy.state_dict(),
                }, model_path + "/" + model_file)
                print("Model Saved!")
                print(f"Directory: {model_path + "/" + model_file}")
                print("\n")

if __name__ == "__main__":
    pass