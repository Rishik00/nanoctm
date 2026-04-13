import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
from torch.utils.data import Dataset
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Components are:
1. Synapse model
2. Backbone
3. Neuron Level models
4. A synchronization engine

5. Dataloader for parity
"""

DROPOUT = 0.01
N_SYNCH_OUT = 32
N_SYNCH_ACTION = 32
NUM_HEADS = 8
N_EMBEDDING = 64
D_EMBEDDING = 128
D_INPUT = 128
D_MODEL = 128
ITERATIONS = 1
SEQUENCE_LENGTH = 64
OUT_DIMS = SEQUENCE_LENGTH * 2
DATASET_SIZE = 100
MEMORY_LENGTH = 10
PREDICTIONS_RESHAPER = [SEQUENCE_LENGTH, 2]
NEURON_SELECT_TYPE = "first-last"

class ParityDataset(Dataset):
    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        vector = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        # print("Initial vector: ", vector)
        vector = vector.float()

        negatives = (vector == -1).to(torch.long)
        # print("Negative: ", negatives)
        cumsum = torch.cumsum(negatives, dim=0)
        # print("Cumsum: ", cumsum)

        target = (cumsum % 2 != 0).to(torch.long)
        # print("Final Target: ", target)
        return vector, target

class Squeeze(nn.Module):
    """
    Squeeze Module.

    Removes a specified dimension of size 1 from the input tensor.
    Useful for incorporating tensor dimension squeezing within nn.Sequential.

    Args:
      dim (int): The dimension to squeeze.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class CustomRotationalEmbedding1D(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding1D, self).__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x):
        start_vector = torch.tensor([0., 1.], device=x.device, dtype=torch.float)
        theta_rad = torch.deg2rad(torch.linspace(0, 180, x.size(2), device=x.device))

        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)

        cos_theta = cos_theta.unsqueeze(1)  # Shape: (height, 1)
        sin_theta = sin_theta.unsqueeze(1)  # Shape: (height, 1)

        # Create rotation matrices
        rotation_matrices = torch.stack([
            torch.cat([cos_theta, -sin_theta], dim=1),
            torch.cat([sin_theta, cos_theta], dim=1)
        ], dim=1)  # Shape: (height, 2, 2)

        # Rotate the start vector
        rotated_vectors = torch.einsum('bij,j->bi', rotation_matrices, start_vector)

        pe = self.projection(rotated_vectors)
        pe = torch.repeat_interleave(pe.unsqueeze(0), x.size(0), 0)
        return pe.transpose(1, 2)

class SuperLinear(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        
        super().__init__()
        # N is the number of neurons (d_model), in_dims is the history length (memory_length)
        self.dropout = nn.Dropout(dropout)
        self.in_dims = in_dims # Corresponds to memory_length

        # LayerNorm applied across the history dimension for each neuron independently
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True)
        self.do_norm = do_norm

        # Initialize weights and biases
        # w1 shape: (memory_length, out_dims, d_model)
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        # b1 shape: (1, d_model, out_dims)
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        # Learnable temperature/scaler T
        self.register_parameter('T', nn.Parameter(torch.Tensor([T]))) 

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape (B, N, in_dims)
                              where B=batch, N=d_model, in_dims=memory_length.
        Returns:
            torch.Tensor: Output tensor, shape (B, N) after squeeze(-1).
        """
        # Input shape: (B, D, M) where D=d_model=N neurons in CTM, M=history/memory length
        out = self.dropout(x)
        # LayerNorm across the memory_length dimension (dim=-1)
        out = self.layernorm(out) # Shape remains (B, N, M)

        # Apply N independent linear models using einsum
        # einsum('BDM,MHD->BDH', ...)
        # x: (B=batch size, D=N neurons, one NLM per each of these, M=history/memory length)
        # w1: (M, H=hidden dims if using MLP, otherwise output, D=N neurons, parallel)
        # b1: (1, D=N neurons, H)
        # einsum result: (B, D, H)
        # Applying bias requires matching shapes, b1 is broadcasted.
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        # Squeeze the output dimension (assumed to be 1 usually) and scale by T
        # This matches the original code's structure exactly.
        out = out.squeeze(-1) / self.T
        return out

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.

    Args:
        tensor: A PyTorch tensor of shape (B, D, H, W).

    Returns:
        A PyTorch tensor of shape (B, D, H, W, 2) with the last dimension
        representing the 2D coordinates within the HW dimensions.
    """
    B, H, W = x.shape

    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)

    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)

    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)

    return coords

def initialize_left_right_neurons(neuron_select_type, synch_type, d_model, n_synch, n_random_pairing_self=0):
    
    if neuron_select_type=='first-last':
        if synch_type == 'out':
            neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
        elif synch_type == 'action':
            neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

    elif neuron_select_type=='random':
        neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

    elif neuron_select_type=='random-pairing':
        assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {neuron_select_type}"
        neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

    device = start_activated_state.device
    return neuron_indices_left.to(device), neuron_indices_right.to(device)


def set_synchronisation_parameters(synch_type: str, n_synch: int, synch_representation_size, n_random_pairing_self: int = 0):
    
    assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
    print("Entering this fn")
    left, right = initialize_left_right_neurons(
        NEURON_SELECT_TYPE,
        synch_type,
        D_MODEL,
        n_synch,
        n_random_pairing_self
    )
    # synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out

    if synch_type == "out":
        out_neuron_indices_left = left
        out_neuron_indices_right = right
        out_decay_params_params = nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True)

        return out_neuron_indices_left, out_neuron_indices_right, out_decay_params_params

    if synch_type == "action":
        action_neuron_indices_left = left
        action_neuron_indices_right = right
        action_decay_params_params = nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True)

        return action_neuron_indices_left, action_neuron_indices_right, action_decay_params_params

def calculate_synch_representation_size(neuron_select_type, n_synch):
    
    if neuron_select_type == 'random-pairing':
        synch_representation_size = n_synch

    elif neuron_select_type in ('first-last', 'random'):
        synch_representation_size = (n_synch * (n_synch + 1)) // 2

    else:
        raise ValueError(f"Invalid neuron selection type: {neuron_select_type}")

    return synch_representation_size

def get_neuron_select_type(neuron_select_type: str):
    
    print(f"Using neuron select type: {neuron_select_type}")
    if neuron_select_type == 'first-last':
        neuron_select_type_out, neuron_select_type_action = 'first', 'last'

    elif neuron_select_type in ('random', 'random-pairing'):
        neuron_select_type_out = neuron_select_type_action = neuron_select_type

    else:
        raise ValueError(f"Invalid neuron selection type: {neuron_select_type}")
    return neuron_select_type_out, neuron_select_type_action

def compute_synchronisation(
    activated_state,
    decay_alpha,
    decay_beta,
    r,
    synch_type,

    n_synch,
    neuron_indices_left,
    neuron_indices_right
):

    if NEURON_SELECT_TYPE in ('first-last', 'random'):
        # For first-last and random, we compute the pairwise sync between all selected neurons
        if NEURON_SELECT_TYPE == 'first-last':
            if synch_type == 'action': # Use last n_synch neurons for action
                selected_left = selected_right = activated_state[:, -n_synch:]
            elif synch_type == 'out': # Use first n_synch neurons for out
                selected_left = selected_right = activated_state[:, :n_synch]
        else: # Use the randomly selected neurons
            selected_left = activated_state[:, neuron_indices_left]
            selected_right = activated_state[:, neuron_indices_right]

        # Compute outer product of selected neurons
        outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
        # Resulting matrix is symmetric, so we only need the upper triangle
        i, j = torch.triu_indices(n_synch, n_synch)
        pairwise_product = outer[:, i, j]

    elif NEURON_SELECT_TYPE == 'random-pairing':
        # For random-pairing, we compute the sync between specific pairs of neurons
        left = activated_state[:, neuron_indices_left]
        right = activated_state[:, neuron_indices_right]
        pairwise_product = left * right
    else:
        raise ValueError("Invalid neuron selection type")

    # Compute synchronisation recurrently
    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product
        decay_beta = torch.ones_like(pairwise_product)
    else:
        decay_alpha = r * decay_alpha + pairwise_product
        decay_beta = r * decay_beta + 1

    synchronisation = decay_alpha / (torch.sqrt(decay_beta))
    return synchronisation, decay_alpha, decay_beta

def compute_normalized_entropy(logits, reduction='mean'):
    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy

def compute_certainty(current_prediction, prediction_reshaper):

    B = current_prediction.size(0)
    reshaped_pred = current_prediction.reshape([B] + prediction_reshaper)
    ne = compute_normalized_entropy(reshaped_pred)

    current_certainty = torch.stack((ne, 1-ne), -1)
    return current_certainty


if __name__ == "__main__":
    
    ## Length is given by DATASET_SIZE and 2 vectors
        # Vector input: SEQUENCE_LENGTH
        # Target: SEQUENCE_LENGTH
    dataset = ParityDataset(SEQUENCE_LENGTH, DATASET_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    vector, target = next(iter(dataloader))
    # print(len(dataset))

    ## MODEL THINGS START HERE
    neuron_select_type_out, neuron_select_type_action = get_neuron_select_type(NEURON_SELECT_TYPE)
    synch_representation_size_action = calculate_synch_representation_size(NEURON_SELECT_TYPE,N_SYNCH_ACTION)
    synch_representation_size_out = calculate_synch_representation_size(NEURON_SELECT_TYPE, N_SYNCH_OUT)

    left_action_indices, right_action_indices, decay_params_action = set_synchronisation_parameters("action", N_SYNCH_ACTION, synch_representation_size_action)
    left_out_indices, right_out_indices, decay_params_out = set_synchronisation_parameters("out", N_SYNCH_OUT, synch_representation_size_out)

    B = vector.size(0)

    # Input is a bunch of -1s, target is a bunch of 1s and 0s
    # vector, target = dataset.__getitem__(0)
    print(target.shape, vector.shape)

    # state_trace is the list of states we see, it's of length MEMORY_LENGTH (makes sense, we iteratively store upto MEMORY_LENGTH)
    start_activated_state = nn.Parameter(torch.zeros(D_MODEL).uniform_())
    state_trace = nn.Parameter(torch.zeros(D_MODEL, MEMORY_LENGTH).uniform_())

    state_trace = state_trace.unsqueeze(0).expand(B, -1, -1)
    activated_state = start_activated_state.unsqueeze(0).expand(B, -1)

    print("Initial state trace: ", state_trace.shape)
    print("Initial activated state: ", start_activated_state.shape)

    predictions = torch.empty(B, OUT_DIMS, ITERATIONS)
    certainities = torch.empty(B, 2, ITERATIONS)

    print("Predictions: ", predictions.shape)
    print("Certainity: ", certainities.shape)

    # Parity backbone according to sakana is an embedding layer.
    parity_backbone = nn.Embedding(N_EMBEDDING, D_EMBEDDING)
    synapses = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.LazyLinear(D_MODEL * 2),
        nn.GLU(),
        nn.LayerNorm(D_MODEL)
    )
    kv_proj = nn.Sequential(
        nn.LazyLinear(D_INPUT),
        nn.LayerNorm(D_INPUT)
    )
    nlm = nn.Sequential(
        nn.Sequential(
            SuperLinear(in_dims=MEMORY_LENGTH, out_dims=2, N=D_MODEL,
                        do_norm=True, dropout=DROPOUT),
            nn.GLU(),
            Squeeze(-1)
        )
    )
    q_proj = nn.LazyLinear(D_INPUT)
    attention = nn.MultiheadAttention(D_INPUT, NUM_HEADS, dropout=DROPOUT, batch_first=True)
    output_projector = nn.Sequential(nn.LazyLinear(OUT_DIMS))
    pos_embedding = CustomRotationalEmbedding1D(D_INPUT)

    # For some reason they do the following
    vector = (vector == 1).long()
    print(vector.shape)

    kv = parity_backbone(vector.long())
    print("KV shape before transpose: ", kv.shape)

    kv = parity_backbone(vector).long().transpose(1, 2)
    print("KV before position embeddings: ", kv.shape)

    pos = pos_embedding(kv)
    print("Positional embedding: ", pos.shape)

    combined = (kv + pos).flatten(2).transpose(1, 2)
    print("Combined: ", combined.shape)

    kv = kv_proj(combined)
    print("Final KV proj shape: ", kv.shape)

    decay_alpha_action, decay_beta_action = None, None
    decay_params_action = torch.clamp(decay_params_action, 0, 15)  # Fix from github user: kuviki
    decay_params_out = torch.clamp(decay_params_out, 0, 15)
    r_action, r_out = torch.exp(-decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-decay_params_out).unsqueeze(0).repeat(B, 1)

    _, decay_alpha_out, decay_beta_out = compute_synchronisation(
        activated_state,
        None, 
        None,
        r_out,
        synch_type="out",

        n_synch=N_SYNCH_OUT,
        neuron_indices_left=left_out_indices,
        neuron_indices_right=right_out_indices
    )

    print("Starting CTM loop: ")
    for i in range(ITERATIONS):
        synchronisation_action, decay_alpha_action, decay_beta_action = compute_synchronisation(
            activated_state,
            decay_alpha_action, 
            decay_beta_action,
            r_action,
            synch_type="action",

            n_synch=N_SYNCH_ACTION,
            neuron_indices_left=left_action_indices,
            neuron_indices_right=right_action_indices
        )
        print("Sync output: ", synchronisation_action.shape)

        q = q_proj(synchronisation_action).unsqueeze(1)
        print("Q proj: ", q.shape)

        attn_out, attn_weights = attention(q, kv, kv, average_attn_weights=False, need_weights=True)
        attn_out = attn_out.squeeze(1)
        print("Attention weight shapes: ", attn_weights.shape)

        pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
        print("Pre synapse input: ", pre_synapse_input.shape)

        state = synapses(pre_synapse_input)
        print("State: ", state.shape)

        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
        print("State trace: ", state_trace.shape)

        activated_state = nlm(state_trace)
        print("Activated state: ", activated_state.shape)

        synchronisation_out, decay_alpha_out, decay_beta_out = compute_synchronisation(
            activated_state, 
            decay_alpha_out, 
            decay_beta_out, 
            r_out, 
            synch_type='out',

            n_synch=N_SYNCH_OUT,
            neuron_indices_left=left_out_indices,
            neuron_indices_right=right_out_indices
        )

        # --- Get Predictions and Certainties ---
        current_prediction = output_projector(synchronisation_out)
        print("prediction shape: ", current_prediction.shape)

        current_certainty = compute_certainty(current_prediction, PREDICTIONS_RESHAPER)

        predictions[..., i] = current_prediction
        certainities[..., i] = current_certainty
        print("Forward pass done!")