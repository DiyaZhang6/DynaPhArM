# /home/zdy/Project2/models/decoder.py

import torch
import torch.nn as nn
from torch_scatter import scatter_mean


# --- Helper Modules ---
class MLP(nn.Module):
    """A simple multi-layer perceptron."""

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
        return self.mlp(x)


class RBFExpansion(nn.Module):
    """Expands a scalar distance into a high-dimensional vector using Gaussian kernels."""

    def __init__(self, d_max=20.0, num_kernels=16, gamma=10.0):
        super().__init__()
        self.centers = torch.linspace(0, d_max, num_kernels)
        self.gamma = gamma

    def forward(self, distances):
        self.centers = self.centers.to(distances.device)
        distances = distances.unsqueeze(-1)
        centers = self.centers.view(1, -1)
        rbf = torch.exp(-self.gamma * (distances - centers) ** 2)
        return rbf


# --- The Decoder Layer ---
class EquivariantDecoderLayer(nn.Module):
    """
    Implements one layer of the SE(3)-Equivariant Graph Transformer Decoder.
    """

    def __init__(self, embed_dim, num_heads, rbf_config, message_mlp_config, coord_weight_mlp_config):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Equivariant Attention Components
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False  # Expects [SeqLen, Batch, Dim]
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 2. Message Passing Components
        self.rbf_expansion = RBFExpansion(
            d_max=rbf_config['d_max'],
            num_kernels=rbf_config['num_kernels']
        )

        message_input_dim = embed_dim * 2 + rbf_config['num_kernels']
        # phi_m: The message MLP
        self.phi_m = MLP(
            in_dim=message_input_dim,
            hidden_dims=message_mlp_config['hidden_dims'],
            out_dim=embed_dim
        )
        self.message_norm = nn.LayerNorm(embed_dim)

        # 3. Coordinate Update Components
        # psi: The coordinate weight prediction MLP
        self.psi = MLP(
            in_dim=embed_dim,  # Input is the message m_ij
            hidden_dims=coord_weight_mlp_config['hidden_dims'],
            out_dim=1  # Output is a scalar weight w_ij
        )

    def forward(self, h, r, batch_index):
        """
        Forward pass for one decoder layer.
        Args:
            h (Tensor): Node scalar features [TotalNodes, D]. Corresponds to h_i^(l-1).
            r (Tensor): Node coordinates [TotalNodes, 3]. Corresponds to r_i^(l-1).
            batch_index (Tensor): Maps each node to its graph in the batch [TotalNodes].
        """
        # --- Subsection: Equivariant Attention ---
        h_res = h
        h_attn_list = []
        for graph_idx in range(batch_index.max() + 1):
            mask = (batch_index == graph_idx)
            if not mask.any(): continue
            # Prepare for MHA: [SeqLen, Batch=1, Dim]
            h_graph = h[mask].unsqueeze(1)
            # Standard self-attention
            h_tilde, _ = self.self_attention(h_graph, h_graph, h_graph, need_weights=False)
            h_attn_list.append(h_tilde.squeeze(1))

        if h_attn_list:
            h_tilde = torch.cat(h_attn_list, dim=0)
            # Update scalar features with residual connection and LayerNorm
            h = self.attn_norm(h_res + h_tilde)

        # --- Subsection: Message Passing & Coordinate Update ---
        num_nodes = h.size(0)
        # Create all pairs for message passing
        pairwise_r = r.unsqueeze(1) - r.unsqueeze(0)
        # d_ij
        d_ij = torch.linalg.norm(pairwise_r, dim=-1) + 1e-8

        # Expand h for pairwise operations
        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)

        # RBF expansion of distances
        rbf_dist = self.rbf_expansion(d_ij)

        # m_ij
        message_input = torch.cat([h_i, h_j, rbf_dist], dim=-1)
        m_ij = self.phi_m(message_input)

        # --- Coordinate Update ---
        # w_ij
        w_ij = self.psi(m_ij).squeeze(-1)  # [N, N]

        # Unit direction vectors
        direction_vecs = pairwise_r / d_ij.unsqueeze(-1)

        # Delta r_i
        # Sum over j for each i
        delta_r = torch.sum(w_ij.unsqueeze(-1) * direction_vecs, dim=1)

        # Update coordinates
        r_new = r + delta_r

        # Also update scalar features with aggregated messages
        h_message_agg = torch.sum(m_ij, dim=1)
        h_new = self.message_norm(h + h_message_agg)

        return h_new, r_new


# --- The Top-Level Decoder Module ---
class StructureDecoder(nn.Module):
    """
    Transforms latent embeddings into a 3D structure using an SE(3)-Equivariant Transformer.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']

        # Initial projection of embeddings
        self.input_projection = MLP(
            in_dim=self.embed_dim,
            hidden_dims=[],
            out_dim=self.embed_dim
        )

        self.layers = nn.ModuleList([
            EquivariantDecoderLayer(
                embed_dim=self.embed_dim,
                num_heads=config['num_heads'],
                rbf_config=config['rbf'],
                message_mlp_config=config['message_mlp'],
                coord_weight_mlp_config=config['coord_weight_mlp']
            ) for _ in range(self.num_layers)
        ])

    def forward(self, e_f_0, r_init, batch_index):
        """
        Args:
            e_f_0 (Tensor): Final diffusion-denoised embeddings [TotalNodes, D].
            r_init (Tensor): Initial coordinates [TotalNodes, 3].
            batch_index (Tensor): Maps each node to its graph in the batch [TotalNodes].

        Returns:
            Tensor: Final predicted coordinates [TotalNodes, 3].
        """
        # Initialize h and r
        h = self.input_projection(e_f_0)
        r = r_init

        # Pass through L decoder layers
        for layer in self.layers:
            h, r = layer(h, r, batch_index)

        # Final coordinate prediction with centering
        # scatter_mean computes the centroid for each graph in the batch
        centroid = scatter_mean(r, batch_index, dim=0)  # [NumGraphs, 3]

        # Broadcast subtraction to center each graph's coordinates
        r_final = r - centroid[batch_index]

        return r_final