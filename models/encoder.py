# /home/zdy/Project2/models/encoder.py

import torch
import torch.nn as nn
import logging
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math


# --- Spherical Harmonics Utility ---
def DifferentiableSphericalHarmonics(vectors: torch.Tensor, lmax: int = 2):
    """
    Computes differentiable spherical harmonics up to l_max=2.
    Args:
        vectors (torch.Tensor): Tensor of shape [..., 3]
        lmax (int): Maximum degree of spherical harmonics.
    Returns:
        torch.Tensor: Tensor of shape [..., (lmax+1)**2]
    """
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    r_sq = x ** 2 + y ** 2 + z ** 2
    r = torch.sqrt(r_sq).unsqueeze(-1) + 1e-8

    sh_features = []
    # l=0
    sh_features.append(torch.full_like(r, 0.28209479177))  # Y_0,0

    if lmax > 0:
        # l=1
        sh_features.append(-0.4886025119 * y / r)  # Y_1,-1
        sh_features.append(0.4886025119 * z / r)  # Y_1,0
        sh_features.append(-0.4886025119 * x / r)  # Y_1,1

    if lmax > 1:
        # l=2
        rr = r_sq.unsqueeze(-1)
        xy = (x * y).unsqueeze(-1)
        yz = (y * z).unsqueeze(-1)
        xz = (x * z).unsqueeze(-1)
        xx_yy = (x ** 2 - y ** 2).unsqueeze(-1)
        zz_rr_3 = (2 * z ** 2 - x ** 2 - y ** 2).unsqueeze(-1)  # Corrected 3*z^2 - r^2

        sh_features.append(0.5 * math.sqrt(3.0 / math.pi) * xy / rr)  # Y_2,-2
        sh_features.append(math.sqrt(3.0 / (4.0 * math.pi)) * yz / rr)  # Y_2,-1, with factor adjustment
        sh_features.append(0.25 * math.sqrt(5.0 / math.pi) * zz_rr_3 / rr)  # Y_2,0
        sh_features.append(math.sqrt(3.0 / (4.0 * math.pi)) * xz / rr)  # Y_2,1, with factor adjustment
        sh_features.append(0.25 * math.sqrt(3.0 / math.pi) * xx_yy / rr)  # Y_2,2

    if lmax > 2:
        logging.warning("Differentiable SH implementation only supports l_max <= 2. Other terms will be zero.")
        num_higher_terms = (lmax + 1) ** 2 - 9
        sh_features.append(torch.zeros(*vectors.shape[:-1], num_higher_terms, device=vectors.device))

    return torch.cat(sh_features, dim=-1)


# --- MLP Utility ---
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation="ReLU", use_batch_norm=False):
        super().__init__()
        activation_map = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}
        act_fn = activation_map.get(activation, nn.ReLU)
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batch_norm: layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x.float())


# --- Encoder Block ---
class CooperativeSE3EncoderBlock(MessagePassing):
    def __init__(self, scalar_dim, vector_dim, edge_scalar_dim,
                 l_max_sh=2, mlp_hidden_dims=[128, 128]):
        super().__init__(aggr='add', node_dim=-2)  # Using simple 'add' aggregation as per algorithm

        self.l_max_sh = l_max_sh
        sh_dim = (l_max_sh + 1) ** 2

        # --- MLPs for NODE UPDATE step ---
        self.edge_vector_mlp = MLP(vector_dim, mlp_hidden_dims, scalar_dim)
        self.message_creation_mlp = MLP(scalar_dim + edge_scalar_dim + scalar_dim, mlp_hidden_dims, scalar_dim)
        self.attention_mlp = MLP(scalar_dim, mlp_hidden_dims, 1)
        self.node_scalar_update_mlp = MLP(scalar_dim + scalar_dim, mlp_hidden_dims, scalar_dim)
        self.node_vector_update_mlp = MLP(scalar_dim, mlp_hidden_dims, vector_dim, activation="SiLU")

        # --- MLPs for EDGE UPDATE step ---
        self.edge_scalar_update_mlp = MLP(edge_scalar_dim + sh_dim, mlp_hidden_dims, edge_scalar_dim)
        self.edge_vector_update_mlp = MLP(vector_dim * 2, mlp_hidden_dims, vector_dim, activation="SiLU")

        # --- MLPs for INTERACTION MODULE ---
        self.interaction_mlp = MLP(scalar_dim, mlp_hidden_dims, scalar_dim)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        """
        Args:
            s (Tensor): Node scalar features [num_nodes, scalar_dim]
            v (Tensor): Node vector features [num_nodes, vector_dim]
            edge_index (Tensor): Edge index [2, num_edges]
            edge_s (Tensor): Edge scalar features [num_edges, edge_scalar_dim]
            edge_v (Tensor): Edge vector features [num_edges, vector_dim]
        Returns:
            Tuple of updated Tensors: (s, v, edge_s, edge_v)
        """
        row, col = edge_index

        # ==========================================================
        # 1. NODE UPDATE STEP
        # ==========================================================

        edge_v_emb = self.edge_vector_mlp(edge_v)
        message_input = torch.cat([s[row], edge_s, edge_v_emb], dim=-1)
        hij = self.message_creation_mlp(message_input)

        # Propagate to get messages at destination nodes, then compute attention
        attention_scores = self.attention_mlp(hij)
        alpha_ij = softmax(attention_scores, col, dim=self.node_dim)

        # scatter_add performs the sum over j in N(i)
        mi = scatter_add(alpha_ij * hij, col, dim=self.node_dim, dim_size=s.size(0))

        # Update node scalar features
        s_update_input = torch.cat([s, mi], dim=-1)
        s_res = self.node_scalar_update_mlp(s_update_input)
        s = s + s_res  # Residual connection

        # Update node vector features
        v_res = self.node_vector_update_mlp(mi)
        v = v + v_res  # Residual connection

        # ==========================================================
        # 2. EDGE UPDATE STEP
        # ==========================================================

        v_diff = v[col] - v[row]  # rij = vj - vi

        h_geometric = DifferentiableSphericalHarmonics(v_diff, lmax=self.l_max_sh)

        edge_s_update_input = torch.cat([edge_s, h_geometric], dim=-1)
        edge_s_res = self.edge_scalar_update_mlp(edge_s_update_input)
        edge_s = edge_s + edge_s_res  # Residual connection

        edge_v_update_input = torch.cat([v[row], v[col]], dim=-1)
        edge_v_res = self.edge_vector_update_mlp(edge_v_update_input)
        edge_v = edge_v + edge_v_res  # Residual connection

        # ==========================================================
        # 3. INTERACTION MODULE
        # ==========================================================
        s_emb_for_interact = self.interaction_mlp(s)
        h_interact = s_emb_for_interact * v  # Element-wise product
        v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
        s = s + v_norm_sq  # Residual boost

        s_norm_sq = torch.sum(s ** 2, dim=-1, keepdim=True)
        v = v + h_interact * s_norm_sq  # Residual boost

        return s, v, edge_s, edge_v


# --- Top-Level Encoder ---
class CooperativeSE3Encoder(nn.Module):
    def __init__(self, in_scalar_dim, in_edge_scalar_dim,
                 hidden_scalar_dim, hidden_vector_dim,
                 num_layers, l_max_sh=2, mlp_hidden_dims=[128, 128]):
        super().__init__()

        # Input projection MLPs to map raw features to the model's hidden dimension
        self.node_scalar_in = MLP(in_scalar_dim, mlp_hidden_dims, hidden_scalar_dim)
        self.edge_scalar_in = MLP(in_edge_scalar_dim, mlp_hidden_dims, hidden_scalar_dim)

        self.layers = nn.ModuleList([
            CooperativeSE3EncoderBlock(
                scalar_dim=hidden_scalar_dim,
                vector_dim=hidden_vector_dim,
                edge_scalar_dim=hidden_scalar_dim,  # Using hidden_scalar_dim for edges too
                l_max_sh=l_max_sh,
                mlp_hidden_dims=mlp_hidden_dims
            )
            for _ in range(num_layers)
        ])

    def forward(self, data):
        """
        Processes a PyTorch Geometric data object.
        Expects:
        - data.node_s: Initial node scalar features
        - data.node_v: Initial node vector features (e.g., coordinates)
        - data.edge_index
        - data.edge_s: Initial edge scalar features
        - data.edge_v: Initial edge vector features (e.g., relative displacement)
        """
        s, v = data.node_s, data.node_v
        edge_s, edge_v = data.edge_s, data.edge_v
        edge_index = data.edge_index

        # 1. Project input features into hidden dimensions
        s = self.node_scalar_in(s)
        edge_s = self.edge_scalar_in(edge_s)

        # 2. Pass through the cooperative encoder blocks
        for layer in self.layers:
            s, v, edge_s, edge_v = layer(s, v, edge_index, edge_s, edge_v)

        # 3. Return the final encoded features
        return s, v, edge_s, edge_v