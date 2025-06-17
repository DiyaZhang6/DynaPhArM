# /home/zdy/Project2/models/encoder.py

import torch
import torch.nn as nn
import logging
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


def DifferentiableSphericalHarmonics(lmax: int, vectors: torch.Tensor, use_differentiable: bool):
    """
    Computes spherical harmonics.
    If use_differentiable is False, it uses the non-differentiable scipy version.
    If True, it uses a differentiable PyTorch implementation (currently supports up to lmax=2).
    """
    if not use_differentiable:
        import scipy.special
        r = torch.linalg.norm(vectors, dim=-1) + 1e-8
        theta = torch.acos(torch.clamp(vectors[..., 2] / r, -1.0, 1.0))
        phi = torch.atan2(vectors[..., 1], vectors[..., 0])

        theta_np = theta.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()

        sh_embeddings = []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                sh = scipy.special.sph_harm(m, l, phi_np, theta_np)
                sh_embeddings.append(torch.from_numpy(sh.real).to(vectors.device).unsqueeze(-1))
        return torch.cat(sh_embeddings, dim=-1).float()

    # Differentiable PyTorch implementation
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
        zz_rr_3 = (3 * z ** 2 - r_sq).unsqueeze(-1)

        sh_features.append(1.09254843059 * xy / rr)  # Y_2,-2
        sh_features.append(-1.09254843059 * yz / rr)  # Y_2,-1
        sh_features.append(0.31539156525 * zz_rr_3 / rr)  # Y_2,0
        sh_features.append(-1.09254843059 * xz / rr)  # Y_2,1
        sh_features.append(0.54627421529 * xx_yy / rr)  # Y_2,2

    if lmax > 2:
        logging.warning("Differentiable SH implementation only supports l_max <= 2. Other terms will be zero.")
        # Pad with zeros for higher orders if requested
        num_higher_terms = (lmax + 1) ** 2 - 9
        sh_features.append(torch.zeros(*vectors.shape[:-1], num_higher_terms, device=vectors.device))

    return torch.cat(sh_features, dim=-1)


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
        return self.mlp(x)


class CooperativeSE3EncoderBlock(MessagePassing):
    def __init__(self, scalar_dim, l_max_sh=2, differentiable_sh=True,
                 mlp_hidden_dims=[128, 128], mlp_activation="ReLU", mlp_use_batch_norm=False):
        super().__init__(aggr=None, node_dim=-2)
        self.l_max_sh = l_max_sh
        self.differentiable_sh = differentiable_sh

        mlp_args = {'activation': mlp_activation, 'use_batch_norm': mlp_use_batch_norm}

        sh_dim = (l_max_sh + 1) ** 2

        self.message_mlp = MLP(scalar_dim * 2 + sh_dim, mlp_hidden_dims, scalar_dim, **mlp_args)
        self.attention_mlp = MLP(scalar_dim, [], 1, **mlp_args)
        self.node_update_mlp = MLP(scalar_dim * 2, mlp_hidden_dims, scalar_dim, **mlp_args)
        self.vector_update_mlp = MLP(scalar_dim, mlp_hidden_dims, 1, **mlp_args)

    def forward(self, h, v, edge_index):
        num_nodes = h.size(0)

        # Vector features (relative positions)
        row, col = edge_index
        v_diff = v[row] - v[col]

        # Spherical harmonics of relative positions
        sh = DifferentiableSphericalHarmonics(self.l_max_sh, v_diff, self.differentiable_sh)

        # Message passing
        m_ij = self.propagate(edge_index, h=h, sh=sh, size=(num_nodes, num_nodes))

        # Node and vector updates
        h_new = self.node_update_mlp(torch.cat([h, m_ij], dim=-1))
        h = h + h_new

        # Equivariant vector update
        vector_weights = self.vector_update_mlp(m_ij).squeeze(-1)
        v_update = scatter_add(v_diff * vector_weights[row].unsqueeze(-1), col, dim=0, dim_size=num_nodes)
        v = v + v_update

        return h, v

    def message(self, h_i, h_j, sh):
        message_input = torch.cat([h_i, h_j, sh], dim=-1)
        m = self.message_mlp(message_input)
        return m

    def aggregate(self, inputs, index, dim_size=None):
        attn_scores = self.attention_mlp(inputs)
        attn_weights = softmax(attn_scores, index, dim=self.node_dim)
        return scatter_add(attn_weights * inputs, index, dim=self.node_dim, dim_size=dim_size)


class CooperativeSE3Encoder(nn.Module):
    def __init__(self, in_scalar_dim, out_scalar_dim, num_layers, **block_kwargs):
        super().__init__()
        self.input_mlp = MLP(in_scalar_dim, [out_scalar_dim], out_scalar_dim)

        self.layers = nn.ModuleList([
            CooperativeSE3EncoderBlock(scalar_dim=out_scalar_dim, **block_kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, s, v, edge_index):
        h = self.input_mlp(s)
        for layer in self.layers:
            h, v = layer(h, v, edge_index)
        return h, v