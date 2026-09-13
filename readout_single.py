"""
The trainable readout.

Everything upstream of this (the eye, flyvis's pretrained visual network,
the central brain connectivity) is real, measured, and frozen. This
module is the one and only piece that learns: a small linear layer
mapping descending neuron activity to a probability distribution over
Tetris actions.

Kept deliberately tiny (a single linear layer) to stay true to the
project's "strict connectome, minimal training" goal, the fly's real
wiring does essentially all the work; this just learns which of its
real escape/steering signals happens to correlate with a good Tetris
move.
"""

import torch
import torch.nn as nn

from tetris_env import NUM_ACTIONS


class ActionReadout(nn.Module):
    def __init__(self, dn_types):
        """
        dn_types: the ordered list of descending neuron type names this
        readout expects as input (from CentralBrainLayer.dn_types), fixes
        the input vector's column order consistently between calls.
        """
        super().__init__()
        self.dn_types = list(dn_types)
        self.linear = nn.Linear(len(self.dn_types), NUM_ACTIONS)

    def activity_dict_to_tensor(self, dn_activity):
        """Converts a {dn_type: value} dict into an ordered tensor,
        matching self.dn_types, filling in 0.0 for anything missing."""
        values = [dn_activity.get(t, 0.0) for t in self.dn_types]
        return torch.tensor(values, dtype=torch.float32)

    def forward(self, dn_activity_tensor):
        """dn_activity_tensor: shape (len(dn_types),). Returns action
        logits, shape (NUM_ACTIONS,)."""
        return self.linear(dn_activity_tensor)

    def act(self, dn_activity_dict):
        """
        Takes a real {dn_type: value} activity dict, returns (action,
        log_prob) sampled from the current policy, for use in the
        training loop.
        """
        x = self.activity_dict_to_tensor(dn_activity_dict)
        logits = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob
