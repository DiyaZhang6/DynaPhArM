# /home/zdy/Project2/models/diffusion.py

import torch
import torch.nn as nn
import math


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


class SinusoidalTimestepEmbedding(nn.Module):
    """Module for creating sinusoidal embeddings for the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionRefiner(nn.Module):
    """
    Refines embeddings using a diffusion process, following standard DDPM principles.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_timesteps = config['num_timesteps']
        schedule_type = config.get('schedule_type', 'linear')
        beta_start = config.get('beta_start', 0.0001)
        beta_end = config.get('beta_end', 0.02)
        denoiser_config = config['denoiser_mlp']
        time_embed_dim = denoiser_config['time_embed_dim']

        # --- Noise Schedule Setup ---
        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif schedule_type == 'cosine':
            steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
            x = steps / self.num_timesteps
            alphas_cumprod = torch.cos(((x + 0.008) / 1.008) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # Register schedule parameters as buffers for device placement
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('recip_sqrt_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1. - torch.cat([torch.tensor([0]), alphas_cumprod[:-1]])) / (1. - alphas_cumprod))

        # --- Denoiser Network (MLP_theta) ---
        self.time_embedding = SinusoidalTimestepEmbedding(time_embed_dim)
        denoiser_input_dim = self.embed_dim + time_embed_dim
        self.denoiser_mlp = MLP(
            in_dim=denoiser_input_dim,
            hidden_dims=denoiser_config['hidden_dims'],
            out_dim=self.embed_dim  # It predicts noise, which has the same dim as the input
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: noise a tensor x_0 to timestep t.
        x_0: [N, D], t: [N] (tensor of timesteps, one per node)
        """
        if noise is None: noise = torch.randn_like(x_0)
        # Gather the correct schedule values for each node's timestep
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        noised_x = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return noised_x

    @torch.no_grad()
    def p_sample(self, x_t, t_per_node, batch_ptr):
        """
        Reverse process: Denoise x_t by one step to get x_{t-1}.
        """
        time_emb = self.time_embedding(t_per_node)
        denoiser_input = torch.cat([x_t, time_emb], dim=-1)
        predicted_noise = self.denoiser_mlp(denoiser_input)

        recip_sqrt_alpha = self.recip_sqrt_alphas[t_per_node].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_per_node].view(-1, 1)
        beta_t = self.betas[t_per_node].view(-1, 1)

        model_mean = recip_sqrt_alpha * (x_t - (beta_t / sqrt_one_minus_alpha_bar) * predicted_noise)

        if t_per_node[0] == 0:
            return model_mean
        else:
            posterior_variance = self.posterior_variance[t_per_node].view(-1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def forward(self, e_f, batch_ptr):
        """
        Full reverse process for inference/refinement.
        Starts with the fused embedding e_f (treated as e_T) and denoises it.
        e_f: [TotalNodes, D], batch_ptr: Pointer for batching.
        """
        x_t = e_f
        for t in reversed(range(self.num_timesteps)):
            num_graphs = len(batch_ptr) - 1
            t_per_graph = torch.full((num_graphs,), t, device=e_f.device, dtype=torch.long)
            t_per_node = t_per_graph.repeat_interleave(torch.diff(batch_ptr))
            x_t = self.p_sample(x_t, t_per_node, batch_ptr)
        return x_t

    def get_training_loss(self, h_all, r_true_all, batch_ptr, noise_pred_head):
        """
        Calculates the training loss for a batch of clean initial embeddings.
        This function orchestrates the steps, but the actual noise prediction
        is done by a head in the main model.
        """
        num_graphs = len(batch_ptr) - 1
        device = h_all.device

        # Sample one timestep per graph in the batch
        t_per_graph = torch.randint(0, self.num_timesteps, (num_graphs,), device=device).long()
        t_per_node = t_per_graph.repeat_interleave(torch.diff(batch_ptr))
        true_coord_noise = torch.randn_like(r_true_all)
        r_t = self.q_sample(r_true_all, t_per_node, true_coord_noise)

        # predict the noise from the feature embeddings h_all
        predicted_coord_noise = noise_pred_head(h_all, r_t, t_per_node)

        # The loss is the MSE between the true and predicted coordinate noise.
        loss = nn.functional.mse_loss(predicted_coord_noise, true_coord_noise)

        return loss