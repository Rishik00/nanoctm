import time
import fire
import math
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from huggingface_hub import PyTorchModelHubMixin
from typing import List
import matplotlib.pyplot as plt

# IMPLICIT ASSUMPTIONS
# neuron_select_types should only be random, random-pairing

class ParityDataset(Dataset):
    def __init__(
        self, 
        sequence_length: int = 75,
        max_samples: int = 100
    ):
        self.sequence_length = sequence_length
        self.max_samples = max_samples

        print(f"Initialized dataset with length: {self.max_samples} with sequence length set to {self.sequence_length}")
    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        # Task at hand: parity. For a given binary sequence the model has to predict the number of even/odd 1s
        # The data is simply generated via cumsum. 

        vector = (torch.randint(0, 2, (self.sequence_length)) - 1).float()
        assert vector.dtype == torch.float32, "Not dtype == float32"

        negatives = (vector == 1).to(torch.long)
        cumsum = torch.sum(negatives, dim=0)

        target = (cumsum%2!=0).to(torch.long)
        return vector, target

class NLM(nn.Module):
    def __init__(self, temperature: float, num_neurons: int, in_dims: int, out_dims: int, dropout: float):
        self.dropout = nn.Dropout(dropout)
        self.in_dims = in_dims

        self.ln = nn.LayerNorm(in_dims, elementwise_affine=True)
        self.register_parameter(
            'w1', nn.Parameter(
                torch.empty((in_dims, out_dims, num_neurons).uniform_(
                    -1/math.sqrt(in_dims+out_dims),
                    -1/math.sqrt(in_dims+out_dims)
                ), requires_grad=True)
            )
        )

        self.register_parameter('b1', nn.Parameter(torch.zeros((1, num_neurons, out_dims)), requires_grad=True))
        self.register_parameter('temperature', nn.Parameter([temperature]))
        self.glu = nn.GLU()

    def forward(self, x: torch.Tensor):
        # types checks and shapes validation
        x = self.dropout(x)
        x = self.ln(x)

        # Do the einsum thing here, or do the loop. 
        x = None
        out = x.squeeze(-1) / self.T

        x = self.glu(x)
        x= x.squeeze(-1)

        return out

class SynapseNet(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        self.synapse_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.LazyLinear(d_model + 2),
            nn.GLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor):
        # type checks and shapes checks
        return self.synapse_net(x)

def get_sync_representation_size(neuron_type: str, num_sync: int):
    # Wonder what the difference between random-pairing and random actually is. 
    if neuron_type == "random-pairing":
        representation_size=num_sync

    elif neuron_type == "ranom-pairing":
        # Why use different ones? 
        representation_size=(num_sync * (num_sync + 1)) // 2

    return representation_size

# Not entirely sure what this function does. 
def init_synchronization_params(neuron_type: str, sync_type: str, model_dim: int, num_sync: int, num_random_pairing_self: int, sync_rep_size: int):
    if neuron_type == 'random':
        indices_left = torch.from_numpy(np.random.choice(np.arange(model_dim), size=num_sync))
        indices_right = torch.from_numpy(np.random.choice(np.arange(model_dim), size=num_sync))

    elif neuron_type == 'random-pairing':
        assert num_sync > num_random_pairing_self, f"Need atleast {num_random_pairing_self} pairs to do something."
        indices_left = torch.from_numpy(np.random.choice(np.arange(model_dim), size=num_sync))
        indices_right = torch.from_numpy(np.random.choice(np.arange(model_dim), size=num_sync))

    # Final device check and assertions before passing back. 
    if sync_type == "out":
        out_decay_params = nn.Parameter(torch.zeros(sync_rep_size), requires_grad=True)
        return indices_left, indices_right, out_decay_params

    elif sync_type == "action":
        action_decay_params = nn.Parameter(torch.zeros(sync_rep_size), requires_grad=True)
        return indices_left, indices_right, action_decay_params

    return None, None, None


# Left overs
def sync():
    pass

def entropy_loss():
    pass

def certainity():
    pass


class NanoCTM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, neuron_type: str, model_dim: int, num_embeddings: int, embedding_dim: int, model_dim: int, num_heads: int, dropout: int):
        self.neuron_type = neuron_type
        self.nlm = NLM()
        self.backbone = nn.Embedding(num_embeddings, embedding_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout, batch_first=True)
        self.pos_embed = None

        self.get_neuron_select_type()
        self.synch_representation_size_action = get_sync_representation_size(self.neuron_type,N_SYNCH_ACTION)
        self.synch_representation_size_out = get_sync_representation_size(self.neuron_type, N_SYNCH_OUT)

        self.left_action_indices, self.right_action_indices, self.decay_params_action = init_synchronization_params("action", N_SYNCH_ACTION, self.synch_representation_size_action)
        self.left_out_indices, self.right_out_indices, self.decay_params_out = init_synchronization_params("out", N_SYNCH_OUT, self.synch_representation_size_out)

        # Activated trace

    def get_neuron_select_type(self):
        
        print(f"Using neuron select type: {self.neuron_type}")
        if self.neuron_type == 'first-last':
            self.neuron_select_type_out, self.neuron_select_type_action = 'first', 'last'

        elif self.neuron_type in ('random', 'random-pairing'):
            self.neuron_select_type_out = self.neuron_select_type_action = self.neuron_type

        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_type}")

    def forward(self, x: torch.Tensor):
        # DType checks

        # Shape checks

        B = x.size(0)
        x = (x == 1).long()

        kv_vector = self.backbone(x)



def train(
    num_samples: int,
    memory_length: int, 
    dropout: float,
    n_sync: int,
    num_embedding: int, 
    embedding_dim: int,
    input_dim: int, 
    output_dim: int,
    model_dim: int, 

    neuron_type: str,
    sync_type: str
):
    print("Initialising CTM run")

if __name__ == "__main__":
    fire.fie(train)