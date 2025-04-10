import torch
import torch.nn as nn
from mRNNTorch.mRNN import mRNN

class RNNPolicy(nn.Module):
    def __init__(
            self, 
            inp_size,
            hid_size,
            output_dim, 
            activation_name="softplus",
            noise_level_act=0.01, 
            noise_level_inp=0.01, 
            constrained=True, 
            dt=10,
            t_const=100,
            batch_first=True,
            lower_bound_rec=0,
            upper_bound_rec=10,
            lower_bound_inp=0,
            upper_bound_inp=10,
            device="cpu"
        ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = 1
        self.constrained = constrained
        self.device = device
        self.dt = dt
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation_name
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp

        self.mrnn = mRNN(
            activation=activation_name,
            noise_level_act=noise_level_act, 
            noise_level_inp=noise_level_inp, 
            constrained=constrained, 
            dt=dt,
            tau=t_const,
            batch_first=batch_first,
            lower_bound_rec=lower_bound_rec,
            upper_bound_rec=upper_bound_rec,
            lower_bound_inp=lower_bound_inp,
            upper_bound_inp=upper_bound_inp,
            device=device
        )

        # Add Region
        self.mrnn.add_recurrent_region("region", hid_size, learnable_bias=True)
        # Add Input
        self.mrnn.add_input_region("input", inp_size)

        # Add connections
        self.mrnn.add_recurrent_connection("region", "region")
        self.mrnn.add_input_connection("input", "region")

        # Finalize connectivity
        self.mrnn.finalize_connectivity()

        self.fc = torch.nn.Linear(self.mrnn.total_num_units, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs, *args, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(h, obs[:, None, :], *args, noise=noise)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(h)).squeeze(dim=1)
        return x, h, u


class GRUPolicy(nn.Module):
    def __init__(
            self, 
            inp_size,
            hid_size,
            output_dim, 
            batch_first=True,
            device="cpu"
        ):
        super().__init__()

        self.output_dim = output_dim
        self.device = device
        self.batch_first = batch_first

        self.gru = nn.GRU(inp_size, hid_size, batch_first=batch_first)

        self.fc = torch.nn.Linear(self.mrnn.total_num_units, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs):
        # Forward pass through mRNN
        x, h = self.mrnn(obs[:, None, :], h)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(h)).squeeze(dim=1)
        return x, h, u