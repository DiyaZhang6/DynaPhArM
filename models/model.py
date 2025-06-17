# /home/zdy/Project2/models/model.py

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

# Import all the building blocks for the model
from .encoder import CooperativeSE3Encoder
from .interaction import InteractionModule, MLP
from .diffusion import DiffusionRefiner
from .decoder import StructureDecoder


class DynaModel(nn.Module):
    """
    Assembles all components (encoder, interaction, diffusion, decoder)
    into an end-to-end model. This model is compatible with the refactored
    data loading pipeline that uses pre-computed graphs.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # --- Encoders for each component ---
        # The input dimensions are calculated based on the feature generation scripts.
        self.backbone_encoder = CooperativeSE3Encoder(
            in_scalar_dim=len(config['backbone_graph_task']['allowed_amino_acids']) + 2,
            out_scalar_dim=config['interaction_params']['embed_dim'],
            **config['backbone_encoder_params']
        )

        sidechain_in_dim = sum([
            len(config['sidechain_graph_task']['atom_symbols']),
            1, 1, 1, 1, 1, 1, 1, 1,
            len(config['sidechain_graph_task']['hybridization_types'])
        ])
        self.sidechain_encoder = CooperativeSE3Encoder(
            in_scalar_dim=sidechain_in_dim,
            out_scalar_dim=config['interaction_params']['embed_dim'],
            **config['sidechain_encoder_params']
        )

        drug_in_dim = sum([
            len(config['drug_graph_task']['atom_symbols']),
            1,
            len(config['drug_graph_task']['hybridization_types']),
            len(config['drug_graph_task']['chiral_tags']),
            1, 1, 1, 1, 1
        ])
        self.drug_encoder = CooperativeSE3Encoder(
            in_scalar_dim=drug_in_dim,
            out_scalar_dim=config['interaction_params']['embed_dim'],
            **config['drug_encoder_params']
        )

        # --- Core Interaction and Refinement Modules ---
        self.interaction_module = InteractionModule(config['interaction_params'])
        self.diffusion_refiner = DiffusionRefiner(config['diffusion_params'])
        self.decoder = StructureDecoder(config['decoder_params'])

        # --- Output Heads ---
        model_cfg = config['model_params']
        embed_dim = config['interaction_params']['embed_dim']

        noise_pred_input_dim = embed_dim
        self.noise_prediction_head = MLP(
            noise_pred_input_dim,
            model_cfg['noise_mlp_hidden_dims'],
            3
        )

        if model_cfg.get('predict_affinity', False):
            self.affinity_head = MLP(
                embed_dim,
                model_cfg['affinity_mlp_hidden_dims'],
                1
            )

    def forward(self, batch: Batch, use_diffusion_refinement=False):
        """
        The main forward pass of the model.
        Args:
            batch (Batch): A PyG HeteroData batch object.
            use_diffusion_refinement (bool): If True, runs the full diffusion reverse process for inference.
        """
        # --- 1. Encode each component ---
        h_b, _ = self.backbone_encoder(
            s=batch['backbone'].node_s,
            v=batch['backbone'].node_v['ca_coord'],
            edge_index=batch['backbone'].edge_index
        )

        if 'sidechain' in batch.node_types and batch['sidechain'].num_nodes > 0:
            h_s, _ = self.sidechain_encoder(
                s=batch['sidechain'].node_s,
                v=batch['sidechain'].node_v,
                edge_index=batch['sidechain'].edge_index
            )
        else:
            h_s = torch.empty(0, self.config['interaction_params']['embed_dim'], device=h_b.device)

        h_d, _ = self.drug_encoder(
            s=batch['drug'].node_s,
            v=batch['drug'].node_v,
            edge_index=batch['drug'].edge_index
        )

        # --- 2. Perform physics-constrained interaction ---
        h_f_b, h_f_s, h_f_d = self.interaction_module(
            h_b, h_s, h_d,
            batch.get(('backbone', 'interacts_with', 'sidechain'), {}).get('s_matrix'),
            batch.get(('backbone', 'interacts_with', 'drug'), {}).get('s_matrix'),
            batch.get(('sidechain', 'interacts_with', 'drug'), {}).get('s_matrix')
        )

        # --- 3. Fuse features and sort coordinates ---
        h_fused_list = [h_f_b, h_f_s, h_f_d]
        h_all = torch.cat([h for h in h_fused_list if h.numel() > 0], dim=0)

        group_ids = batch.atom_group_ids
        sorted_indices = torch.argsort(group_ids)

        r_init_all = batch.r_init[sorted_indices]
        r_true_all = batch.r_true[sorted_indices]

        batch_index_list = [batch[nt].batch for nt in ['backbone', 'sidechain', 'drug'] if nt in batch.node_types]
        batch_index = torch.cat(batch_index_list, dim=0)

        # --- 4. Diffusion and Decoding ---
        if use_diffusion_refinement:
            # Inference Mode: Run the full reverse diffusion process to refine features
            h_refined = self.diffusion_refiner(h_all, batch.ptr)
        else:
            # Training Mode: Use the fused features directly
            h_refined = h_all

        pred_coords = self.decoder(h_refined, r_init_all, batch_index)

        # --- 5. Noise Prediction for Training Loss ---
        r_t, true_noise, _ = self.diffusion_refiner.get_training_outputs(h_all, r_true_all, batch.ptr)
        pred_noise = self.noise_prediction_head(h_all)

        # --- 6. Affinity Prediction ---
        pred_affinity = None
        if hasattr(self, 'affinity_head'):
            global_features = scatter_mean(h_refined, batch_index, dim=0)
            pred_affinity = self.affinity_head(global_features).squeeze(-1)

        return {
            "pred_coords": pred_coords,
            "true_coords": r_true_all,
            "pred_noise": pred_noise,
            "true_noise": true_noise,
            "pred_affinity": pred_affinity,
            "true_affinity": batch.affinity
        }