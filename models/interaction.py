# /home/zdy/Project2/models/interaction.py


import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PhysicsConstrainedCrossAttention(nn.Module):
    """Implements the core physics-constrained attention mechanism."""

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.w_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, head_dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(1))
        self.scale = head_dim ** -0.5

    def forward(self, query_embed, key_embed, value_embed, s_matrix):
        """Handles PyG-style batched inputs."""
        q = self.w_q(query_embed)
        k = self.w_k(key_embed)
        v = self.w_v(value_embed)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        physics_constrained_scores = (attn_scores * self.scale) + (self.gamma * s_matrix)
        attention_weights = F.softmax(physics_constrained_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output


class InteractionModule(nn.Module):
    """
    Computes interactions between components.
    """

    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config['embed_dim']
        num_heads = config['num_heads']
        mlp_hidden_dims = config.get('mlp_hidden_dims', [512])
        mlp_activation = config.get('mlp_activation', 'ReLU')

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # ModuleDict for all 6 interaction directions
        self.attentions = nn.ModuleDict({
            's_to_b': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
            'd_to_b': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
            'b_to_s': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
            'd_to_s': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
            'b_to_d': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
            's_to_d': nn.ModuleList(
                [PhysicsConstrainedCrossAttention(embed_dim, self.head_dim) for _ in range(num_heads)]),
        })

        # Fusion MLPs for each component.
        # Input: [Incoming message 1, Incoming message 2, Outgoing context]
        mlp_in_dim = embed_dim * 3
        self.mlp_b = MLP(mlp_in_dim, mlp_hidden_dims, embed_dim, activation=mlp_activation)
        self.mlp_s = MLP(mlp_in_dim, mlp_hidden_dims, embed_dim, activation=mlp_activation)
        self.mlp_d = MLP(mlp_in_dim, mlp_hidden_dims, embed_dim, activation=mlp_activation)

    def _run_multihead_attention(self, heads, query, key, value, s_matrix):
        """Helper to run multi-head attention and concatenate results."""
        if s_matrix is None or query.numel() == 0 or key.numel() == 0:
            return torch.zeros(query.shape[0], self.embed_dim, device=query.device)
        return torch.cat([h(query, key, value, s_matrix) for h in heads], dim=-1)

    def forward(self, h_b, h_s, h_d, s_bs, s_bd, s_sd):
        """
        Processes PyG-style batched inputs and performs the modified fusion.
        """
        # --- 1. Compute all 6 directional attention outputs ---
        # Messages targeting Backbone
        out_s_to_b = self._run_multihead_attention(self.attentions['s_to_b'], h_b, h_s, h_s, s_bs)
        out_d_to_b = self._run_multihead_attention(self.attentions['d_to_b'], h_b, h_d, h_d, s_bd)

        # Messages targeting Sidechain
        out_b_to_s = self._run_multihead_attention(self.attentions['b_to_s'], h_s, h_b, h_b,
                                                   s_bs.transpose(-1, -2) if s_bs is not None else None)
        out_d_to_s = self._run_multihead_attention(self.attentions['d_to_s'], h_s, h_d, h_d, s_sd)

        # Messages targeting Drug
        out_b_to_d = self._run_multihead_attention(self.attentions['b_to_d'], h_d, h_b, h_b,
                                                   s_bd.transpose(-1, -2) if s_bd is not None else None)
        out_s_to_d = self._run_multihead_attention(self.attentions['s_to_d'], h_d, h_s, h_s,
                                                   s_sd.transpose(-1, -2) if s_sd is not None else None)

        # --- 2. Create summary of outgoing messages for each component ---

        # Summary of influence FROM b (avg of messages received by s and d from b)
        if out_b_to_s.numel() > 0 and out_b_to_d.numel() > 0:
            context_from_b = torch.cat([out_b_to_s, out_b_to_d], dim=0).mean(dim=0, keepdim=True)
        elif out_b_to_s.numel() > 0:
            context_from_b = out_b_to_s.mean(dim=0, keepdim=True)
        elif out_b_to_d.numel() > 0:
            context_from_b = out_b_to_d.mean(dim=0, keepdim=True)
        else:
            context_from_b = torch.zeros(1, self.embed_dim, device=h_b.device)

        # Summary of influence FROM s
        if out_s_to_b.numel() > 0 and out_s_to_d.numel() > 0:
            context_from_s = torch.cat([out_s_to_b, out_s_to_d], dim=0).mean(dim=0, keepdim=True)
        else:  # Handle cases with no drug or no backbone
            context_from_s = torch.zeros(1, self.embed_dim, device=h_s.device)

        # Summary of influence FROM d
        if out_d_to_b.numel() > 0 and out_d_to_s.numel() > 0:
            context_from_d = torch.cat([out_d_to_b, out_d_to_s], dim=0).mean(dim=0, keepdim=True)
        else:
            context_from_d = torch.zeros(1, self.embed_dim, device=h_d.device)

        # --- 3. Fuse embeddings using incoming messages and outgoing context ---
        # Update backbone features
        # Expand context to match the number of backbone nodes
        context_b_expanded = context_from_b.expand(h_b.shape[0], -1)
        fused_b = self.mlp_b(torch.cat([out_s_to_b, out_d_to_b, context_b_expanded], dim=-1))
        h_f_b = h_b + fused_b

        # Update sidechain features
        if h_s.numel() > 0:
            context_s_expanded = context_from_s.expand(h_s.shape[0], -1)
            fused_s = self.mlp_s(torch.cat([out_b_to_s, out_d_to_s, context_s_expanded], dim=-1))
            h_f_s = h_s + fused_s
        else:
            h_f_s = h_s

        # Update drug features
        context_d_expanded = context_from_d.expand(h_d.shape[0], -1)
        fused_d = self.mlp_d(torch.cat([out_b_to_d, out_s_to_d, context_d_expanded], dim=-1))
        h_f_d = h_d + fused_d

        return h_f_b, h_f_s, h_f_d