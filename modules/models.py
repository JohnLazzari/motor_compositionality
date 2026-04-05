import torch
import torch.nn as nn
from mrnntorch import mRNN


class RNNPolicy(nn.Module):
    def __init__(
        self,
        inp_size,
        hid_size,
        output_dim,
        activation_name="softplus",
        noise_level_act=0.01,
        noise_level_inp=0.01,
        rec_constrained=False,
        inp_constrained=False,
        dt=10,
        t_const=100,
        batch_first=True,
        device="cpu",
        add_new_rule_inputs=False,
        num_new_inputs=10,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = 1
        self.rec_constrained = rec_constrained
        self.inp_constrained = rec_constrained
        self.device = device
        self.dt = dt
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation_name

        self.mrnn = mRNN(
            activation=activation_name,
            noise_level_act=noise_level_act,
            noise_level_inp=noise_level_inp,
            rec_constrained=rec_constrained,
            inp_constrained=inp_constrained,
            dt=dt,
            tau=t_const,
            batch_first=batch_first,
            device=device,
        )

        # Add Region
        self.mrnn.add_recurrent_region(
            "region", hid_size, learnable_bias=True, device=device
        )
        # Add Input
        self.mrnn.add_input_region("input", inp_size, device=device)

        # Add connections
        self.mrnn.add_recurrent_connection("region", "region")
        self.mrnn.add_input_connection("input", "region")

        """
            Will replace original input region. This allows to load all other parameters but make new inputs
            This can probably be done much easier by simply adding an input region with 2 inputs
        """
        if add_new_rule_inputs:
            self.mrnn.add_input_region("input_new_rules", num_new_inputs, device=device)
            self.mrnn.add_input_connection("input_new_rules", "region")

            self.mrnn.add_input_region("condition", inp_size - 10, device=device)
            self.mrnn.add_input_connection("condition", "region")

            self.mrnn.inp_dict["condition"].connections["region"]["parameter"].data = (
                self.mrnn.inp_dict["input"]
                .connections["region"]["parameter"]
                .data[:, 10:]
            )

            # Since were just deleting make sure this is in check
            self.mrnn.total_num_inputs = inp_size - 10 + num_new_inputs

            del self.mrnn.inp_dict["input"]

        # Finalize connectivity
        self.mrnn.finalize_connectivity()

        self.fc = torch.nn.Linear(self.mrnn.total_num_units, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, obs, x, h, *args, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(obs[:, None, :], x, h, *args, noise=noise)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        x = x.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(h)).squeeze(dim=1)
        return x, h, u


class GRUPolicy(nn.Module):
    def __init__(self, inp_size, hid_size, output_dim, batch_first=True, device="cpu"):
        super().__init__()

        self.output_dim = output_dim
        self.device = device
        self.batch_first = batch_first

        self.gru = nn.GRU(inp_size, hid_size, batch_first=batch_first)

        self.fc = torch.nn.Linear(hid_size, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs):
        # Forward pass through mRNN
        hs, h = self.gru(obs[:, None, :], h[None, :, :])
        # Squeeze in the time dimension (doing timesteps one by one)
        hs = hs.squeeze(1)
        # Motor output
        u = self.sigmoid(self.fc(hs)).squeeze(dim=1)
        return None, hs, u
