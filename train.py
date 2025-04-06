import os
import sys
import torch
import motornet as mn
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Policy
from losses import l1_dist, l1_rate, l1_weight, l1_muscle_act
from envs import RandomReach, DlyRandomReach, Maze
from utils import save_hp, create_dir

DEF_HP = {
    "activation_name": "softplus",
    "noise_level_act": 0.1,
    "noise_level_inp": 0.01,
    "constrained": False,
    "dt": 10,
    "t_const": 20,
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 50_000,
    "save_iter": 100,
    "l1_rate": 0.001,
    "l1_weight": 0.001,
    "l1_muscle_act": 0.001
}

def train_2link(config_path, model_path, model_file, hp=None):

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

    policy = Policy(
        config_path, 
        effector.n_muscles, 
        activation_name=hp["activation_name"],
        noise_level_act=hp["noise_level_act"], 
        noise_level_inp=hp["noise_level_inp"], 
        constrained=hp["constrained"], 
        dt=hp["dt"],
        t_const=hp["t_const"],
        device=device
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    losses = []
    interval = 10

    env_list = [RandomReach, DlyRandomReach, Maze]
    probs = [1/3, 1/3, 1/3]

    for batch in range(hp["epochs"]):

        # initialize batch
        h = torch.zeros(size=(hp["batch_size"], policy.mrnn.total_num_units))
        h = policy.mrnn.get_initial_condition(h)

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
            x, h, action = policy(h, obs)
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
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], sum(losses[-interval:])/interval))

        """
        if batch % hp["save_iter"] == 0:
            torch.save({
                'agent_state_dict': policy.state_dict(),
            }, model_path + "/" + model_file)
        """

if __name__ == "__main__":
    pass