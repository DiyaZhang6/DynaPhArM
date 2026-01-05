# /home/zdy/Project2/scripts/data.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import List, Optional, Dict

try:
    from torch_geometric.data import HeteroData, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    print("FATAL: PyTorch Geometric is not installed. Please install it.")
    exit(1)


class ProteinLigandGraphDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-computed graphs for backbone, sidechain, and drug,
    combines them with physics-geometry labels and Lennard-Jones matrices,
    and structures everything into a single PyTorch Geometric `HeteroData` object.
    """

    def __init__(self, pdb_ids: list, config: dict):
        super().__init__()
        self.pdb_ids = pdb_ids
        project_root_str = config.get('project_base_dir', '.')
        self.project_root = Path(project_root_str).resolve()

        # Define paths from config
        self.graph_dir = self.project_root / config['backbone_graph_task']['output_dir']  # Graph dir is shared
        self.labels_dir = self.project_root / config['data_loading']['labels_dir']
        self.lj_dir = self.project_root / config['lj_generation_task']['output_dir']

        logging.info(
            f"Dataset Initialized. Graph dir: {self.graph_dir}, Labels dir: {self.labels_dir}, LJ dir: {self.lj_dir}")

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> Optional[HeteroData]:
        pdb_id = self.pdb_ids[idx]

        graph_subdir = self.graph_dir / pdb_id
        label_subdir = self.labels_dir / pdb_id
        lj_subdir = self.lj_dir / pdb_id

        files = {
            'backbone_graph': graph_subdir / f"{pdb_id}_backbone.pt",
            'sidechain_graphs': graph_subdir / f"{pdb_id}_sidechain.pt",
            'drug_graph': graph_subdir / f"{pdb_id}_drug.pt",
            'labels': label_subdir / f"{pdb_id}_labels.pt",
            's_bs': lj_subdir / 'bs.npy',
            's_bd': lj_subdir / 'bd.npy',
            's_sd': lj_subdir / 'sd.npy'
        }

        required_files = ['backbone_graph', 'drug_graph', 'labels', 's_bs', 's_bd', 's_sd']
        if not all(files[key].exists() for key in required_files):
            logging.warning(f"Missing one or more required files for {pdb_id}. Skipping.")
            return None

        try:
            backbone_data = torch.load(files['backbone_graph'], map_location='cpu')
            drug_data = torch.load(files['drug_graph'], map_location='cpu')
            phy_geo_labels = torch.load(files['labels'], map_location='cpu')
            sidechain_graphs = torch.load(files['sidechain_graphs'], map_location='cpu') if files[
                'sidechain_graphs'].exists() else []

            data = HeteroData()

            # 1. Populate Backbone Nodes
            data['backbone'].node_s = backbone_data['node_s']
            data['backbone'].node_v = backbone_data['node_v']
            data['backbone'].edge_index = backbone_data['edge_index']
            data['backbone'].edge_s = backbone_data['edge_s']
            data['backbone'].edge_v = backbone_data['edge_v']
            data['backbone'].residue_ids = backbone_data['residue_ids']

            # 2. Populate Sidechain Nodes by concatenating all residue sidechains
            if sidechain_graphs:
                sc_node_s, sc_node_v, sc_edge_s, sc_edge_idx = [], [], [], []
                sc_atom_res_map = []  # Map each sidechain atom back to its residue index in the backbone graph
                current_atom_idx = 0

                # Create a map from residue ID tuple to its index in the backbone graph
                bb_res_id_to_idx = {tuple(res_id): i for i, res_id in enumerate(backbone_data['residue_ids'])}

                for sc_graph in sidechain_graphs:
                    num_atoms = sc_graph['node_s'].shape[0]
                    sc_node_s.append(sc_graph['node_s'])
                    sc_node_v.append(sc_graph['node_v_coords'])
                    sc_edge_s.append(sc_graph['edge_s'])

                    if sc_graph['edge_index'].numel() > 0:
                        sc_edge_idx.append(sc_graph['edge_index'] + current_atom_idx)

                    # Map sidechain atoms to the corresponding backbone residue index
                    res_idx_in_bb = bb_res_id_to_idx.get(tuple(sc_graph['residue_id']))
                    if res_idx_in_bb is not None:
                        sc_atom_res_map.extend([res_idx_in_bb] * num_atoms)
                    else:
                        sc_atom_res_map.extend([-1] * num_atoms)

                    current_atom_idx += num_atoms

                data['sidechain'].node_s = torch.cat(sc_node_s, dim=0)
                data['sidechain'].node_v = torch.cat(sc_node_v, dim=0)
                data['sidechain'].edge_index = torch.cat(sc_edge_idx, dim=1) if sc_edge_idx else torch.empty((2, 0),
                                                                                                             dtype=torch.long)
                data['sidechain'].edge_s = torch.cat(sc_edge_s, dim=0)
                data['sidechain'].atom_to_residue_map = torch.tensor(sc_atom_res_map, dtype=torch.long)

            # 3. Populate Drug Nodes
            data['drug'].node_s = drug_data['node_scalar_features']
            data['drug'].node_v = drug_data['atom_coordinates']
            data['drug'].edge_index = drug_data['edge_index']
            data['drug'].edge_s = drug_data['edge_scalar_features']

            # 4. Add Physics/Geometry labels and other top-level attributes
            data.pdb_id = pdb_id
            data.affinity = phy_geo_labels.get('affinity', torch.tensor(float('nan')))
            data.r_true = phy_geo_labels['r_true']
            data.r_init = phy_geo_labels['r_init']
            data.bond_indices = phy_geo_labels['bond']['indices']
            data.ref_bond_lengths = phy_geo_labels['bond']['ref_lengths']
            data.angle_indices = phy_geo_labels['angle']['indices']
            data.ref_angles = phy_geo_labels['angle']['ref_angles']
            data.dihedral_indices = phy_geo_labels['dihedral']['indices']
            data.true_dihedrals = phy_geo_labels['dihedral']['true_angles']
            data.vdw_indices = phy_geo_labels['vdw']['indices']
            data.electro_indices = phy_geo_labels['electro']['indices']
            data.hbond_indices = phy_geo_labels['hbond']['indices']
            data.pi_pi_ring_pair_indices = phy_geo_labels['pi_pi']['ring_pair_indices']
            data.atom_group_ids = phy_geo_labels['atom_group_ids']
            data.partial_charges = phy_geo_labels['partial_charges']
            data.vdw_radii = phy_geo_labels['vdw']['radii']

            # 5. Assign pre-computed LJ matrices
            if 'backbone' in data.node_types and 'sidechain' in data.node_types and files['s_bs'].exists():
                data['backbone', 'interacts_with', 'sidechain'].s_matrix = torch.from_numpy(
                    np.load(files['s_bs'])).float()
            if 'backbone' in data.node_types and 'drug' in data.node_types and files['s_bd'].exists():
                data['backbone', 'interacts_with', 'drug'].s_matrix = torch.from_numpy(np.load(files['s_bd'])).float()
            if 'sidechain' in data.node_types and 'drug' in data.node_types and files['s_sd'].exists():
                data['sidechain', 'interacts_with', 'drug'].s_matrix = torch.from_numpy(np.load(files['s_sd'])).float()

            return data

        except Exception as e:
            logging.error(f"Failed to create graph data for {pdb_id}. Skipping. Error: {e}", exc_info=True)
            return None


def custom_hetero_collate_fn(data_list: List[HeteroData]) -> Batch:
    """
    Custom collate function to handle batching of HeteroData objects,
    specifically creating block-diagonal matrices for `s_matrix` attributes.
    """
    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return Batch()

    batch = Batch.from_data_list(data_list, follow_batch=['atom_to_residue_map'])

    s_matrix_types = [
        ('backbone', 'interacts_with', 'sidechain'),
        ('backbone', 'interacts_with', 'drug'),
        ('sidechain', 'interacts_with', 'drug')
    ]

    for edge_type in s_matrix_types:
        if 's_matrix' in batch.get_edge_store(*edge_type):
            s_matrix_list = batch.get_edge_store(*edge_type).get('s_matrix')
            if s_matrix_list:
                try:
                    block_diag_s_matrix = torch.block_diag(*s_matrix_list)
                    batch.get_edge_store(*edge_type).s_matrix = block_diag_s_matrix
                except Exception as e:
                    logging.error(f"Error in block_diag for {edge_type}: {e}")
                    pass
    return batch


def get_data_loader(config: dict, split: str, split_file_path: str = None) -> Optional[PyGDataLoader]:
    """
    Creates a PyG DataLoader using the new ProteinLigandGraphDataset.
    """
    data_cfg = config['data_loading']
    train_cfg = config.get('training', {})
    project_root = Path(config.get('project_base_dir', '.')).resolve()

    if split_file_path is None:
        split_map = {'train': 'train_split_file', 'val': 'val_split_file'}
        if split not in split_map:
            # Handle test sets from the list in config
            test_sets = {ts['name']: ts['path'] for ts in data_cfg.get('test_sets', [])}
            if split in test_sets:
                split_file_path = test_sets[split]
            else:
                raise ValueError(
                    f"For split '{split}', a direct 'split_file_path' or a corresponding entry in 'test_sets' must be provided.")
        else:
            split_file_path = data_cfg[split_map[split]]

    split_file_full_path = project_root / split_file_path

    try:
        df = pd.read_csv(split_file_full_path, comment='#', header=None)
        pdb_ids = df.iloc[:, 0].tolist()
    except Exception as e:
        logging.error(f"Failed to read or process {split_file_full_path}: {e}")
        return None

    dataset = ProteinLigandGraphDataset(pdb_ids=pdb_ids, config=config)

    # Filter out None values that may have occurred during __getitem__
    dataset.samples = [s for s in dataset if s is not None]

    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', 0)

    data_loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=custom_hetero_collate_fn,
    )

    logging.info(
        f"Created PyG DataLoader for '{split}' split from {Path(split_file_path).name} with {len(dataset)} samples and batch size {batch_size}.")
    return data_loader