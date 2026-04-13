import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from huggingface_hub import PyTorchModelHubMixin
from typing import List
import matplotlib.pyplot as plt

DTYPE = torch.float32
DROPOUT: float = 0.01
N_SYNCH_OUT: int = 32
N_SYNCH_ACTION: int = 32
NUM_HEADS: int = 8
N_EMBEDDING: int = 64
D_EMBEDDING: int= 128
D_INPUT: int = 128
D_MODEL: int = 128
ITERATIONS: int = 1
SEQUENCE_LENGTH: int = 64
OUT_DIMS: int = SEQUENCE_LENGTH * 2
DATASET_SIZE: int = 100
MEMORY_LENGTH: int = 10
PREDICTIONS_RESHAPER: List[int] = [SEQUENCE_LENGTH, 2]

# Major assumption: we're assuming that the neuron types is either: "first-last"/"random"
NEURON_TYPE: str = "first-last"
SYNC_TYPE: str = "out"

class ParityDataset(Dataset):
    def __init__(self, sequence_length=75, length=100000):
        self.sequence_length = sequence_length
        self.dataset_length = length

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        vector = 2 * torch.randint(0,2,(self.sequence_length, )) - 1
        vector = vector.float()

        negatives = (vector == -1).to(torch.long)
        cumsum = torch.sum(negatives, dim=0)

        target = (cumsum % 2 != 0).to(torch.long)
        return vector, target

class ParityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass


class NLMOriginal(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        N, # Number of neurons
        T=1.0,  # Learnable temperature/scaler T
        do_norm=False,
        dropout=0
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_dims = in_dims # Corresponds to memory_length

        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True)
        self.do_norm = do_norm

        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )

        # b1 shape: (1, d_model, out_dims)
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        self.register_parameter('T', nn.Parameter(torch.Tensor([T]))) 

    def forward(self, x):
        out = self.dropout(x)

        out = self.layernorm(out) # Shape remains (B, N, M)
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        out = out.squeeze(-1) / self.T
        return out

class NLMCustom(nn.Module):
    def __init__(
        self, 
        num_neurons: int, 
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1

    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        for i in range(self.num_neurons):
            self.register_parameter(
                f'w_{i}', nn.Parameter(
                    torch.empty((self.input_dim, self.output_dim)).uniform_(
                        -1/math.sqrt(self.input_dim + self.output_dim),
                        1/math.sqrt(self.input_dim + self.output_dim)
                    ), requires_grad=True)
            )

            self.register_parameter(f'b_{i}', nn.Parameter(torch.zeros((1, self.num_neurons, self.output_dim))))

    def forward(self, x: torch.Tensor):
        # x: [B, N, M]  (batch, neurons, history)
        out = self.dropout(x)
        
        results = []
        for i in range(self.num_neurons):
            neuron_input = out[:, i, :]  # [B, M] - this neuron's history
            w = self.get_parameter(f'w_{i}')  # [M, H]
            b = self.get_parameter(f'b_{i}')  # [H]
            neuron_out = neuron_input @ w + b  # [B, H]
            results.append(neuron_out)
        
        output = torch.stack(results, dim=1)  # [B, N, H]
        return output

class SynapseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.LazyLinear(D_MODEL * 2),
            nn.GLU(),
            nn.LayerNorm(D_MODEL)
        )

    def forward(self, x):
        # Shape check
        return self.net(x)

def init_left_right_neurons(device):
    if NEURON_TYPE == "first-last":
        if SYNC_TYPE == "out":
            neuron_left, neuron_right = torch.arange(0, N_SYNCH_OUT)
        elif SYNC_TYPE == "action":
            neuron_left, neuron_right = torch.arange(0, N_SYNCH_ACTION)

    elif NEURON_TYPE == "random":
        if SYNC_TYPE == "action":
            neuron_left = torch.from_numpy(np.random.choice(np.arange(D_MODEL), size=N_SYNCH_ACTION))
            neuron_right = torch.from_numpy(np.random.choice(np.arange(D_MODEL), size=N_SYNCH_ACTION))
        elif SYNC_TYPE == "out":
            neuron_left = torch.from_numpy(np.random.choice(np.arange(D_MODEL), size=N_SYNCH_ACTION))
            neuron_right = torch.from_numpy(np.random.choice(np.arange(D_MODEL), size=N_SYNCH_ACTION))
    
    return neuron_left.to(device), neuron_right.to(device)


def init_sync(
    synch_representation_size, 
    device
):
    left, right = init_left_right_neurons(device)

    if SYNC_TYPE == "out":
        out_neuron_indices_left = left
        out_neuron_indices_right = right
        out_decay_params_params = nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True)

        return out_neuron_indices_left, out_neuron_indices_right, out_decay_params_params

    if SYNC_TYPE == "action":
        action_neuron_indices_left = left
        action_neuron_indices_right = right
        action_decay_params_params = nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True)

        return action_neuron_indices_left, action_neuron_indices_right, action_decay_params_params



def sync(
    astate: torch.Tensor,
    decay_alpha,
    decay_beta,
    r,
    neurons_left,
    neurons_right
):
    # For first-last and random, we compute the pairwise sync between all selected neurons
    if NEURON_TYPE == 'first-last':
        if SYNC_TYPE == 'action': # Use last n_synch neurons for action
            selected_left = selected_right = astate[:, -N_SYNCH_ACTION:]
        elif SYNC_TYPE == 'out': # Use first N_SYNCH_ACTION neurons for out
            selected_left = selected_right = astate[:, :N_SYNCH_OUT]
    else: # Use the randomly selected neurons
        selected_left = astate[:, neurons_left]
        selected_right = astate[:, neurons_right]

    # Compute outer product of selected neurons
    outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
    # Resulting matrix is symmetric, so we only need the upper triangle
    i, j = torch.triu_indices(N_SYNCH_ACTION, N_SYNCH_ACTION)
    pairwise_product = outer[:, i, j]

    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product
        decay_beta = torch.ones_like(pairwise_product)
    else:
        decay_alpha = r * decay_alpha + pairwise_product
        decay_beta = r * decay_beta + 1

    synchronisation = decay_alpha / (torch.sqrt(decay_beta))
    return synchronisation, decay_alpha, decay_beta


def entropy_loss(logits, reduction='mean'):
    preds = torch.softmax(logits)
    num_classes = preds.shape[-1]

    max_entropy = torch.log(torch.tensor(num_classes, dtype=DTYPE))
    entropy = -torch.sum(
        preds * torch.log_softmax(
            logits, dim=-1)
        , dim=-1
    )

    normalized_entropy = entropy / max_entropy

    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy

# Not exactly a loss but you get the point
def certainity_loss(current_prediction):
    B = current_prediction.size(0)
    reshaped_pred = current_prediction.reshape([B] + PREDICTIONS_RESHAPER)
    normalized_entropy = entropy_loss(reshaped_pred)

    certainity = torch.stack((normalized_entropy, 1-normalized_entropy), -1)
    return certainity
    
class NanoCTM(nn.Module, PyTorchModelHubMixin):
    def __init__(self) -> None:
        super(NanoCTM, self).__init__()

        self.neuron_level_model = NLMOriginal(
            in_dims=MEMORY_LENGTH, 
            out_dims=2, 
            d_model=D_MODEL, 
            do_norm=True, 
            dropout=DROPOUT
        )
        self.positional_embedding = None

        self.q_proj = nn.LazyLinear(D_INPUT)
        self.kv_proj = nn.Sequential(
            nn.LazyLinear(D_INPUT), 
            nn.LazyLinear(D_INPUT)
        )
        self.attention = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)


        self.left_action_indices, self.right_action_indices, self.decay_params_action = init_sync("action", N_SYNCH_ACTION, synch_representation_size_action)
        self.left_out_indices, self.right_out_indices, self.decay_params_out = init_sync("out", N_SYNCH_OUT, synch_representation_size_out)

        self.activated_state = self.register_parameter(nn.Parameter())
        self.start_state = self.register_parameter(nn.Parameter())

    def forward(self, x):
        batch_size: int = x.size(0)
        vector, target = x

        print("vector and target shapes: ", vector.shape, target.shape)

        for i in range(ITERATIONS):
            pass



if __name__ == "__main__":
    dataset = ParityDataset(SEQUENCE_LENGTH, DATASET_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

