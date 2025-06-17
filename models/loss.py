#!/usr/bin/env python
# /home/zdy/Project2/models/loss.py
# Implements the physics and geometry-guided loss functions for protein-ligand complex generation,
# with support for dynamic, adaptive loss weighting.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# --- Helper Function for Pi-Pi Stacking ---

def _calculate_pi_pi_geometry(pred_coords: torch.Tensor, ring_pair_indices: List):
    """
    Calculates pi-pi stacking geometric parameters (centroids, normals, distance, etc.)
    on-the-fly from predicted coordinates and pre-identified ring atom indices.
    This function is fully differentiable.
    """
    if not ring_pair_indices:
        return {
            'distances': torch.tensor([], device=pred_coords.device),
            'angles_rad': torch.tensor([], device=pred_coords.device),
            'displacements': torch.tensor([], device=pred_coords.device),
            'normals1': torch.tensor([], device=pred_coords.device),
            'normals2': torch.tensor([], device=pred_coords.device)
        }

    centroids1, centroids2 = [], []
    normals1, normals2 = [], []

    for r1_indices, r2_indices in ring_pair_indices:
        # Move indices to the same device as coordinates
        r1_indices = r1_indices.to(pred_coords.device)
        r2_indices = r2_indices.to(pred_coords.device)

        r1_coords = pred_coords[r1_indices]
        r2_coords = pred_coords[r2_indices]

        # Calculate centroids
        c1 = torch.mean(r1_coords, dim=0)
        c2 = torch.mean(r2_coords, dim=0)
        centroids1.append(c1)
        centroids2.append(c2)

        # Calculate normal vectors (robust to 5 or 6-membered rings)
        v1_1 = r1_coords[1] - r1_coords[0]
        v1_2 = r1_coords[-1] - r1_coords[0]
        n1 = F.normalize(torch.cross(v1_1, v1_2), p=2, dim=-1)

        v2_1 = r2_coords[1] - r2_coords[0]
        v2_2 = r2_coords[-1] - r2_coords[0]
        n2 = F.normalize(torch.cross(v2_1, v2_2), p=2, dim=-1)

        normals1.append(n1)
        normals2.append(n2)

    centroids1 = torch.stack(centroids1)
    centroids2 = torch.stack(centroids2)
    normals1 = torch.stack(normals1)
    normals2 = torch.stack(normals2)

    # Calculate distance between centroids
    vec_c1_c2 = centroids2 - centroids1
    distances = torch.linalg.norm(vec_c1_c2, dim=-1)

    # Calculate angle between normal vectors
    cos_theta = torch.sum(normals1 * normals2, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angles_rad = torch.acos(cos_theta)

    # Calculate lateral displacement
    # Displacement of C2 projected onto the plane of Ring 1
    proj_on_n1 = torch.sum(vec_c1_c2 * normals1, dim=-1).unsqueeze(-1) * normals1
    displacements = torch.linalg.norm(vec_c1_c2 - proj_on_n1, dim=-1)

    return {
        'distances': distances,
        'angles_rad': angles_rad,
        'displacements': displacements,
        'normals1': normals1,
        'normals2': normals2
    }


# --- Individual Physics and Geometry Loss Components ---

def loss_bond_length(pred_lengths, ref_lengths):
    """Computes the bond length deviation loss."""
    if pred_lengths.numel() == 0: return torch.tensor(0.0, device=pred_lengths.device)
    ref_lengths = torch.clamp(ref_lengths, min=1e-6)
    loss = ((pred_lengths - ref_lengths) / ref_lengths) ** 2
    return torch.sum(loss)


def loss_bond_angle(pred_angles_rad, ref_angles_rad):
    """Computes the bond angle deviation loss."""
    if pred_angles_rad.numel() == 0: return torch.tensor(0.0, device=pred_angles_rad.device)
    loss = (pred_angles_rad - ref_angles_rad) ** 2
    return torch.sum(loss)


def loss_dihedral_angle(pred_angles_rad, true_angles_rad):
    """Computes the dihedral angle deviation loss."""
    if pred_angles_rad.numel() == 0: return torch.tensor(0.0, device=pred_angles_rad.device)
    diff = pred_angles_rad - true_angles_rad
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.sum(diff ** 2)


def loss_vdw_repulsion(pred_coords, pair_indices, atom_radii):
    """Computes the van der Waals repulsion loss for steric clashes."""
    if pair_indices.numel() == 0: return torch.tensor(0.0, device=pred_coords.device)
    vectors = pred_coords[pair_indices[:, 1]] - pred_coords[pair_indices[:, 0]]
    distances = torch.linalg.norm(vectors, dim=-1)
    # Get VdW radii sum for each pair
    radii_sum = atom_radii[pair_indices[:, 0]] + atom_radii[pair_indices[:, 1]]
    violations = radii_sum - distances
    return torch.sum(F.relu(violations) ** 2)


def loss_electrostatic_coulomb(pred_coords, partial_charges, pair_indices, params: Dict):
    """Computes the Coulombic electrostatic loss based on point charges."""
    if pair_indices.numel() == 0: return torch.tensor(0.0, device=pred_coords.device)
    q_i = partial_charges[pair_indices[:, 0]]
    q_j = partial_charges[pair_indices[:, 1]]
    r_i = pred_coords[pair_indices[:, 0]]
    r_j = pred_coords[pair_indices[:, 1]]

    r_ij_sq = torch.sum((r_j - r_i) ** 2, dim=-1)
    r_ij_smooth = torch.sqrt(r_ij_sq + params['smoothing_delta'] ** 2)

    energy = (params['C_unit_conv'] * q_i * q_j) / (params['dielectric_constant'] * r_ij_smooth)
    # Penalize large energy terms
    return torch.sum(energy ** 2)


def loss_hydrogen_bond(pred_coords, hbond_triplets, params: Dict):
    """Computes the hydrogen bond geometry loss using parameters from config."""
    if hbond_triplets.numel() == 0: return torch.tensor(0.0, device=pred_coords.device)

    d_coords, h_coords, a_coords = pred_coords[hbond_triplets[:, 0]], pred_coords[hbond_triplets[:, 1]], pred_coords[
        hbond_triplets[:, 2]]

    # Distance loss (penalizes D-A distance outside ideal range)
    da_dist = torch.linalg.norm(d_coords - a_coords, dim=-1)
    loss_dist = F.relu(da_dist - params['dist_max']) ** 2 + F.relu(params['dist_min'] - da_dist) ** 2

    # Angle loss (penalizes D-H-A angle outside ideal range)
    vec_hd = d_coords - h_coords
    vec_ha = a_coords - h_coords
    vec_hd_norm = F.normalize(vec_hd, p=2, dim=-1)
    vec_ha_norm = F.normalize(vec_ha, p=2, dim=-1)
    dot = torch.sum(vec_hd_norm * vec_ha_norm, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta_rad = torch.acos(dot)
    theta_deg = torch.rad2deg(theta_rad)

    loss_angle = F.relu(params['angle_min_deg'] - theta_deg) ** 2 + F.relu(theta_deg - params['angle_max_deg']) ** 2

    return torch.sum(loss_dist + loss_angle)


def loss_pi_pi_stacking(pred_geom: Dict, params: Dict):
    """Computes pi-pi stacking loss from on-the-fly calculated geometry and config parameters."""
    if pred_geom['distances'].numel() == 0:
        return torch.tensor(0.0, device=pred_geom['distances'].device)

    pred_distances = pred_geom['distances']
    pred_angles_rad = pred_geom['angles_rad']
    pred_displacements = pred_geom['displacements']
    pred_normals1 = pred_geom['normals1']
    pred_normals2 = pred_geom['normals2']

    # --- Geometric Loss (L_pi_pi_geo) ---
    loss_dist_upper = F.relu(pred_distances - params['dist_max']) ** 2
    loss_dist_lower = F.relu(params['dist_min'] - pred_distances) ** 2
    loss_angle = F.relu(torch.rad2deg(pred_angles_rad) - params['angle_max_deg']) ** 2
    loss_disp = F.relu(pred_displacements - params['disp_max']) ** 2
    L_pi_pi_geo = torch.sum(loss_dist_upper + loss_dist_lower + loss_angle + loss_disp)

    # --- Alignment Loss (L_pi_pi_align) ---
    cos_theta = torch.sum(pred_normals1 * pred_normals2, dim=-1)
    alignment_term = (1 - torch.abs(cos_theta)) ** 2
    distance_gate = F.relu(params['alignment_dist_gate'] - pred_distances)
    L_pi_pi_align = torch.sum(alignment_term * distance_gate)

    return L_pi_pi_geo + L_pi_pi_align


# --- Main Loss Class ---

class TotalLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config['total_loss']
        self.params = self.config['params']
        self.weight_strategy = self.config.get('weight_strategy', 'static')

        if self.weight_strategy == 'dynamic':
            dynamic_config = self.config.get('dynamic_weights', {})
            self.epsilon = dynamic_config.get('epsilon', 1e-8)
            self.smoothing_factor = dynamic_config.get('smoothing_factor', 0.0)

            # Register buffer for EMA of losses.
            self.register_buffer('ema_L_noise', torch.tensor(1.0))
            self.register_buffer('ema_L_structure', torch.tensor(1.0))
            self.register_buffer('ema_L_phy_geo', torch.tensor(1.0))
            self.first_run = True

        elif self.weight_strategy == 'static':
            self.static_weights = self.config['static_weights']
        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")

    def _calculate_dynamic_weights(self, l_noise, l_structure, l_phy_geo):
        # Detach losses to prevent gradients from flowing through the weight calculation
        l_noise_val = l_noise.detach()
        l_structure_val = l_structure.detach()
        l_phy_geo_val = l_phy_geo.detach()

        # Update EMA of losses
        if self.first_run:
            self.ema_L_noise.copy_(l_noise_val)
            self.ema_L_structure.copy_(l_structure_val)
            self.ema_L_phy_geo.copy_(l_phy_geo_val)
            self.first_run = False
        else:
            self.ema_L_noise.copy_(self.smoothing_factor * self.ema_L_noise + (1 - self.smoothing_factor) * l_noise_val)
            self.ema_L_structure.copy_(
                self.smoothing_factor * self.ema_L_structure + (1 - self.smoothing_factor) * l_structure_val)
            self.ema_L_phy_geo.copy_(
                self.smoothing_factor * self.ema_L_phy_geo + (1 - self.smoothing_factor) * l_phy_geo_val)

        # Use the (smoothed) EMA values for weight calculation
        inv_L_noise = 1.0 / (self.ema_L_noise + self.epsilon)
        inv_L_structure = 1.0 / (self.ema_L_structure + self.epsilon)
        inv_L_phy_geo = 1.0 / (self.ema_L_phy_geo + self.epsilon)

        sum_inv_L = inv_L_noise + inv_L_structure + inv_L_phy_geo

        # Normalize to get weights that sum to 1
        w_noise = inv_L_noise / sum_inv_L
        w_structure = inv_L_structure / sum_inv_L
        w_phy_geo = inv_L_phy_geo / sum_inv_L

        return w_noise, w_structure, w_phy_geo

    def forward(self, pred_coords, true_coords, pred_noise, true_noise, phy_geo_labels: Dict):
        # --- Calculate individual loss components ---
        L_noise = F.mse_loss(pred_noise, true_noise)
        L_structure = F.mse_loss(pred_coords, true_coords)

        # --- Calculate Physics-Geometry Loss (L_phy_geo) ---
        # Covalent losses
        if 'bond' in phy_geo_labels and phy_geo_labels['bond']['indices'].numel() > 0:
            bond_vectors = pred_coords[phy_geo_labels['bond']['indices'][:, 1]] - pred_coords[
                phy_geo_labels['bond']['indices'][:, 0]]
            pred_bond_lengths = torch.linalg.norm(bond_vectors, dim=-1)
            L_bond = loss_bond_length(pred_bond_lengths, phy_geo_labels['bond']['ref_lengths'])
        else:
            L_bond = torch.tensor(0.0, device=pred_coords.device)

        if 'angle' in phy_geo_labels and phy_geo_labels['angle']['indices'].numel() > 0:
            angle_indices = phy_geo_labels['angle']['indices']
            vec_ji = pred_coords[angle_indices[:, 0]] - pred_coords[angle_indices[:, 1]]
            vec_jk = pred_coords[angle_indices[:, 2]] - pred_coords[angle_indices[:, 1]]
            dot_product = torch.sum(F.normalize(vec_ji, p=2, dim=-1) * F.normalize(vec_jk, p=2, dim=-1), dim=-1)
            pred_bond_angles_rad = torch.acos(dot_product.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            L_angle = loss_bond_angle(pred_bond_angles_rad, phy_geo_labels['angle']['ref_angles'])
        else:
            L_angle = torch.tensor(0.0, device=pred_coords.device)

        if 'dihedral' in phy_geo_labels and phy_geo_labels['dihedral']['indices'].numel() > 0:
            dih_indices = phy_geo_labels['dihedral']['indices']
            p0, p1, p2, p3 = pred_coords[dih_indices[:, 0]], pred_coords[dih_indices[:, 1]], pred_coords[
                dih_indices[:, 2]], pred_coords[dih_indices[:, 3]]
            b0, b1, b2 = -1.0 * (p1 - p0), p2 - p1, p3 - p2
            b1_norm = F.normalize(b1, p=2, dim=-1)
            n1 = F.normalize(torch.cross(b0, b1_norm), p=2, dim=-1)
            n2 = F.normalize(torch.cross(b1_norm, b2), p=2, dim=-1)
            m1 = torch.cross(n1, b1_norm)
            x, y = torch.sum(n1 * n2, dim=-1), torch.sum(m1 * n2, dim=-1)
            pred_dihedral_angles_rad = torch.atan2(y, x)
            L_dihedral = loss_dihedral_angle(pred_dihedral_angles_rad, phy_geo_labels['dihedral']['true_angles'])
        else:
            L_dihedral = torch.tensor(0.0, device=pred_coords.device)

        # Non-covalent losses
        L_vdw = loss_vdw_repulsion(pred_coords, phy_geo_labels['vdw']['indices'], phy_geo_labels['vdw']['radii'])
        L_electro = loss_electrostatic_coulomb(pred_coords, phy_geo_labels['partial_charges'],
                                               phy_geo_labels['electro']['indices'],
                                               self.params['electrostatic'])
        L_hbond = loss_hydrogen_bond(pred_coords, phy_geo_labels['hbond']['indices'], self.params['hbond'])

        pred_pi_pi_geom = _calculate_pi_pi_geometry(pred_coords, phy_geo_labels['pi_pi']['ring_pair_indices'])
        L_pi_pi = loss_pi_pi_stacking(pred_pi_pi_geom, self.params['pi_pi'])

        L_phy_geo = L_bond + L_angle + L_dihedral + L_vdw + L_electro + L_hbond + L_pi_pi

        # --- Calculate Total Loss with selected weighting strategy ---
        if self.weight_strategy == 'dynamic':
            w_noise, w_structure, w_phy_geo = self._calculate_dynamic_weights(L_noise, L_structure, L_phy_geo)
            L_total = w_noise * L_noise + w_structure * L_structure + w_phy_geo * L_phy_geo
        else:  # 'static'
            w_noise = self.static_weights['L_noise']
            w_structure = self.static_weights['L_structure']
            w_phy_geo = self.static_weights['L_phy_geo']
            L_total = w_noise * L_noise + w_structure * L_structure + w_phy_geo * L_phy_geo

        return {
            'total_loss': L_total, 'L_noise': L_noise, 'L_structure': L_structure, 'L_phy_geo': L_phy_geo,
            'L_bond': L_bond, 'L_angle': L_angle, 'L_dihedral': L_dihedral, 'L_vdW': L_vdw,
            'L_electro': L_electro, 'L_hbond': L_hbond, 'L_pi_pi': L_pi_pi,
            'w_noise': w_noise, 'w_structure': w_structure, 'w_phy_geo': w_phy_geo  # Log weights
        }