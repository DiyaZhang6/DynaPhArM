#!/usr/bin/env python
# phy_geo.py
# Pre-processes high-quality complex files to extract ground truth labels (r_true, physics, atom_groups),
# and combines them with initial coordinates (r_init) from docked structures.

import os
import argparse
import logging
import logging.handlers
import yaml
import datetime
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Optional

# --- Third-party imports ---
try:
    from schrodinger.structure import StructureReader, Structure
    from schrodinger import utility as schrodinger_utility
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pandas as pd
except ImportError:
    print("FATAL: This script requires Schrodinger's Python API, RDKit, and pandas.")
    print("Please run this script in a Conda environment with these packages installed.")
    exit(1)

# --- Global Config ---
CONFIG = None
PROJECT_ROOT = None
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}  # Define backbone atoms for grouping


# --- Configuration & Logging ---
def setup_logging(log_config_key: str):
    """Sets up logging based on the configuration."""
    global CONFIG, PROJECT_ROOT
    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    log_level = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', True)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{log_base_name}_{timestamp}.log"
    else:
        log_file_name = f"{log_base_name}.log"

    log_path = Path(log_dir_rel)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path = log_path / log_file_name

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.getLevelName(log_level))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info(f"Logging for '{log_config_key}' configured. Level: {log_level}. Output file: {log_path}")


def load_config(config_path: Path):
    """Loads the YAML config and sets global project root."""
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: CONFIG = yaml.safe_load(f)
    PROJECT_ROOT = config_path.parent.resolve()


# --- Data Loading Abstraction ---

def load_affinity_data(file_paths: list[str]) -> dict[str, float]:
    """Loads affinity data from multiple CSV/space-delimited files."""
    affinity_map = {}
    if not file_paths:
        logging.warning("No affinity files specified in the config. No affinity labels will be generated.")
        return affinity_map

    logging.info("Loading affinity data from specified files...")
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logging.warning(f"Affinity data file not found: {path}. Skipping.")
            continue
        try:
            df = pd.read_csv(path, sep=r'\s+|,', engine='python', header=None, comment='#', usecols=[0, 1],
                             names=['pdb_id', 'affinity'], on_bad_lines='warn')
            df.dropna(subset=['pdb_id', 'affinity'], inplace=True)
            for _, row in df.iterrows():
                try:
                    affinity_map[str(row['pdb_id'])] = float(row['affinity'])
                except (ValueError, TypeError):
                    continue
            logging.info(
                f"Loaded {len(df)} valid entries from {path}. Total unique entries so far: {len(affinity_map)}")
        except Exception as e:
            logging.error(f"Failed to read or process affinity data from {path}: {e}")

    logging.info(f"Successfully compiled a map of {len(affinity_map)} unique affinity entries.")
    return affinity_map


def load_from_maegz(file_path: Path) -> Tuple[Optional[Structure], Optional[str]]:
    """Loads a ground truth complex from a .maegz file."""
    try:
        st = next(StructureReader(str(file_path)))
        pdb_id = file_path.stem.split('_')[0]
        return st, pdb_id
    except StopIteration:
        logging.warning(f"No structures found in {file_path}. Skipping.")
        return None, None
    except Exception as e:
        logging.error(f"Failed to load maegz file {file_path}: {e}")
        return None, None


def load_from_pdb_sdf_pair(dir_path: Path) -> Tuple[Optional[Structure], Optional[str]]:
    """Loads a ground truth complex from a protein.pdb and ligand.sdf pair."""
    pdb_id = dir_path.name
    try:
        protein_path = dir_path / f"{pdb_id}_protein.pdb"
        ligand_path = dir_path / f"{pdb_id}_ligand.sdf"

        if not protein_path.exists():
            logging.warning(f"Missing protein file: {protein_path}. Skipping.")
            return None, None
        if not ligand_path.exists():
            logging.warning(f"Missing ligand file: {ligand_path}. Skipping.")
            return None, None

        protein_st = next(StructureReader(str(protein_path)))
        ligand_st = next(StructureReader(str(ligand_path)))
        complex_st = protein_st.merge(ligand_st)
        return complex_st, pdb_id
    except StopIteration:
        logging.warning(f"Could not read structure for {pdb_id}. Skipping.")
        return None, None
    except Exception as e:
        logging.error(f"Failed to load PDB/SDF pair for {pdb_id}: {e}", exc_info=False)
        return None, None


def load_initial_coords_from_split(pdb_id: str, task_config: dict) -> Optional[torch.Tensor]:
    """
    Loads initial coordinates (r_init) from the output of split.py.
    """
    init_struct_dir = task_config.get('initial_structure_dir')
    if not init_struct_dir:
        return None

    base_dir = Path(init_struct_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir

    dir_path = base_dir / pdb_id
    if not dir_path.is_dir():
        return None

    templates = task_config.get('split_file_templates', {})
    backbone_path = dir_path / f"{pdb_id}{templates.get('backbone_suffix', '_backbone.pdb')}"
    sidechain_path = dir_path / f"{pdb_id}{templates.get('sidechain_suffix', '_sidechain.pdb')}"
    ligand_path = dir_path / f"{pdb_id}{templates.get('ligand_suffix', '_ligand.sdf')}"

    try:
        if not all(p.exists() for p in [backbone_path, sidechain_path, ligand_path]):
            logging.debug(f"One or more initial component files missing for {pdb_id}. Will use noise fallback.")
            return None

        backbone_st = next(StructureReader(str(backbone_path)))
        sidechain_st = next(StructureReader(str(sidechain_path)))
        ligand_st = next(StructureReader(str(ligand_path)))

        # Merge and extract coordinates ONLY
        init_complex = backbone_st.merge(sidechain_st).merge(ligand_st)
        coords = np.array([atom.xyz for atom in init_complex.atom], dtype=np.float32)
        return torch.from_numpy(coords)

    except Exception as e:
        logging.warning(f"Could not load initial coords for {pdb_id}: {e}. Will use noise fallback.")
        return None


DATA_LOADERS = {
    "maegz_complex": load_from_maegz,
    "pdb_sdf_pair": load_from_pdb_sdf_pair,
}


def get_atom_groups_from_structure(st: Structure) -> torch.Tensor:
    """
    Identifies which atoms belong to backbone, sidechain, or ligand.

    Args:
        st (Structure): The input Schrodinger Structure object.

    Returns:
        torch.Tensor: A tensor of group IDs (0: backbone, 1: sidechain, 2: ligand/drug).
    """
    group_ids = []
    for atom in st.atom:
        residue = atom.getResidue()

        if residue.is_protein:
            if atom.pdbname.strip() in _BACKBONE_ATOMS:
                group_ids.append(0)  # Backbone
            else:
                group_ids.append(1)  # Sidechain
        else:  # Assumed to be ligand, water, ion, etc.
            group_ids.append(2)  # Ligand/Drug

    return torch.tensor(group_ids, dtype=torch.long)


# --- Feature Extraction Functions ---

def st_to_rdkit_mol(st: 'schrodinger.structure.Structure'):
    """Converts a Schrodinger Structure object to an RDKit Mol object, extracts charges and atomic numbers."""
    with schrodinger_utility.NamedTemporaryFile(suffix=".sdf") as tmp_sdf:
        st.write(tmp_sdf.name, format='sdf')
        mol_supplier = Chem.SDMolSupplier(str(tmp_sdf.name), removeHs=False, sanitize=False)
        if not mol_supplier: return None, None, None
        mol = mol_supplier[0]
        if mol:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                             catchErrors=True)
            partial_charges = [atom.partial_charge for atom in st.atom]
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            return mol, torch.tensor(partial_charges, dtype=torch.float32), torch.tensor(atomic_nums, dtype=torch.long)
    return None, None, None


def get_vdw_radii(mol, config: dict):
    """Gets van der Waals radii for each atom from config."""
    vdw_radii_map = config['vdw_radii']
    default_radius = config['default_vdw_radius']
    radii = [vdw_radii_map.get(atom.GetSymbol(), default_radius) for atom in mol.GetAtoms()]
    return torch.tensor(radii, dtype=torch.float32)


def get_topology_and_references(mol, config: dict):
    """Extracts topology (bonds, angles, dihedrals) and their reference values from config."""
    bonds, angles, dihedrals = [], [], []
    ref_bonds, ref_angles, true_dihedrals_rad = [], [], []

    default_bond_len = config['default_bond_length']
    default_angle_rad = np.deg2rad(config['default_bond_angle_deg'])

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            ff = AllChem.MMFFGetMoleculeForceField(mol)
            ff_type = 'MMFF94'
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            ff_type = 'UFF'
    except Exception:
        ff = None
        ff_type = 'None'

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append(sorted((i, j)))
        if ff:
            params = ff.GetMMFFBondStretchParams(mol, i, j) if ff_type == 'MMFF94' else ff.GetUFFBondStretchParams(mol,
                                                                                                                   i, j)
            ref_bonds.append(params[0] if params else default_bond_len)
        else:
            ref_bonds.append(default_bond_len)

    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        if atom_i.GetDegree() >= 2:
            neighbors = [n.GetIdx() for n in atom_i.GetNeighbors()]
            for j_idx in range(len(neighbors)):
                for k_idx in range(j_idx + 1, len(neighbors)):
                    j, k = neighbors[j_idx], neighbors[k_idx]
                    angles.append(sorted((j, i, k)))
                    if ff:
                        params = ff.GetMMFFAngleBendParams(mol, j, i,
                                                           k) if ff_type == 'MMFF94' else ff.GetUFFAngleBendParams(mol,
                                                                                                                   j, i,
                                                                                                                   k)
                        ref_angles.append(np.deg2rad(params[0]) if params else default_angle_rad)
                    else:
                        ref_angles.append(default_angle_rad)

    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(j).GetDegree() > 1 and mol.GetAtomWithIdx(k).GetDegree() > 1:
            for ni in mol.GetAtomWithIdx(j).GetNeighbors():
                i = ni.GetIdx()
                if i == k: continue
                for nl in mol.GetAtomWithIdx(k).GetNeighbors():
                    l = nl.GetIdx()
                    if l == j: continue
                    if len({i, j, k, l}) < 4: continue
                    dihedrals.append((i, j, k, l))
                    true_dihedrals_rad.append(AllChem.GetDihedralRad(mol.GetConformer(), i, j, k, l))

    return {
        'bonds': torch.tensor(bonds, dtype=torch.long), 'ref_lengths': torch.tensor(ref_bonds, dtype=torch.float32),
        'angles': torch.tensor(angles, dtype=torch.long), 'ref_angles': torch.tensor(ref_angles, dtype=torch.float32),
        'dihedrals': torch.tensor(dihedrals, dtype=torch.long),
        'true_angles': torch.tensor(true_dihedrals_rad, dtype=torch.float32)
    }


def get_ring_info(mol, config: dict):
    """Identifies aromatic rings and computes their centroids, normals, and ATOM INDICES."""
    aromatic_rings = []
    positions = mol.GetConformer().GetPositions()
    norm_thresh = config['ring_normal_norm_threshold']

    for ring_indices_tuple in mol.GetRingInfo().AtomRings():
        ring_indices = list(ring_indices_tuple)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_indices):
            ring_pos = positions[ring_indices]
            centroid = np.mean(ring_pos, axis=0)
            v1 = ring_pos[1] - ring_pos[0]
            v2 = ring_pos[-1] - ring_pos[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > norm_thresh:
                normal /= norm
            else:
                normal = np.array([0., 0., 1.])
            aromatic_rings.append({'centroid': centroid, 'normal': normal, 'indices': ring_indices})
    return aromatic_rings


def get_interaction_labels(mol, topology, config: dict):
    """Identifies pairs and computes values for non-covalent interactions using config."""
    num_atoms = mol.GetNumAtoms()
    all_pairs = torch.combinations(torch.arange(num_atoms), r=2)
    bonded_set = {tuple(sorted(p)) for p in topology['bonds'].tolist()}
    angle_set = {tuple(sorted((p[0], p[2]))) for p in topology['angles'].tolist()}
    non_bonded_mask = torch.tensor(
        [tuple(sorted(p.tolist())) not in bonded_set and tuple(sorted(p.tolist())) not in angle_set for p in all_pairs])

    hbond_triplets = []
    donor_s = Chem.MolFromSmarts(config['hbond_donors_smarts'])
    acceptor_s = Chem.MolFromSmarts(config['hbond_acceptors_smarts'])
    donors_idx = sum(mol.GetSubstructMatches(donor_s), ())
    acceptors_idx = sum(mol.GetSubstructMatches(acceptor_s), ())
    for d_idx in donors_idx:
        d_atom = mol.GetAtomWithIdx(d_idx)
        for h_atom in d_atom.GetNeighbors():
            if h_atom.GetAtomicNum() == 1:
                for a_idx in acceptors_idx:
                    if d_idx != a_idx:
                        hbond_triplets.append((d_idx, h_atom.GetIdx(), a_idx))

    rings = get_ring_info(mol, config)
    pi_pi_ring_pair_indices = []
    pi_pi_dist_cutoff = config['pi_pi_distance_cutoff']

    if len(rings) >= 2:
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                ring1, ring2 = rings[i], rings[j]
                d_ij = np.linalg.norm(ring1['centroid'] - ring2['centroid'])
                if d_ij <= pi_pi_dist_cutoff:
                    pi_pi_ring_pair_indices.append(
                        [torch.tensor(ring1['indices'], dtype=torch.long),
                         torch.tensor(ring2['indices'], dtype=torch.long)])
    return {
        'non_bonded_pairs': all_pairs[non_bonded_mask],
        'hbond_triplets': torch.tensor(hbond_triplets, dtype=torch.long),
        'pi_pi_ring_pair_indices': pi_pi_ring_pair_indices,
    }


# --- Main Processing Logic ---
def process_entry(ground_truth_st: Structure, pdb_id: str, task_config: dict, affinity_map: dict):
    """
    Processes a single entry to create a label file containing both
    ground truth data (from crystal structure) and initial data (from docked structure).
    """
    output_path = Path(task_config['output_dir'])
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_file = output_path / pdb_id / f"{pdb_id}_labels.pt"

    if not task_config.get('overwrite_existing', False) and output_file.exists():
        logging.info(f"Skipping {pdb_id}: Output file already exists.")
        return 'skipped'

    try:
        # --- 1. Process GROUND TRUTH structure for labels ---
        mol_true, partial_charges, atomic_nums = st_to_rdkit_mol(ground_truth_st)
        if not mol_true:
            logging.error(f"Failed to convert ground truth structure to RDKit mol for {pdb_id}")
            return 'error'

        # Get atom groups FROM THE GROUND TRUTH STRUCTURE
        atom_group_ids = get_atom_groups_from_structure(ground_truth_st)

        r_true = torch.tensor(mol_true.GetConformer().GetPositions(), dtype=torch.float32)
        vdw_radii = get_vdw_radii(mol_true, task_config)
        topology = get_topology_and_references(mol_true, task_config)
        interactions = get_interaction_labels(mol_true, topology, task_config)
        affinity_value = affinity_map.get(pdb_id)

        # --- 2. Load INITIAL COORDS from docked structure ---
        r_init = load_initial_coords_from_split(pdb_id, task_config)

        # --- 3. Validate and Finalize ---
        if r_init is not None:
            if r_init.shape != r_true.shape:
                logging.error(
                    f"Shape mismatch for {pdb_id}! True: {r_true.shape}, Init: {r_init.shape}. Using noise fallback.")
                r_init = None  # Invalidate if shapes don't match

        if r_init is None:
            # create r_init by adding noise to r_true
            noise_scale = task_config.get('fallback_noise_scale', 1.0)
            noise = torch.randn_like(r_true) * noise_scale
            r_init = r_true + noise

        # --- 4. Assemble final labels dictionary ---
        labels = {
            'r_true': r_true,
            'r_init': r_init,
            'atomic_nums': atomic_nums,
            'atom_group_ids': atom_group_ids,
            'partial_charges': partial_charges,
            'bond': {'indices': topology['bonds'], 'ref_lengths': topology['ref_lengths']},
            'angle': {'indices': topology['angles'], 'ref_angles': topology['ref_angles']},
            'dihedral': {'indices': topology['dihedrals'], 'true_angles': topology['true_angles']},
            'vdw': {'indices': interactions['non_bonded_pairs'], 'radii': vdw_radii},
            'hbond': {'indices': interactions['hbond_triplets']},
            'pi_pi': {'ring_pair_indices': interactions['pi_pi_ring_pair_indices']},
            'electro': {'indices': interactions['non_bonded_pairs']}
        }

        if affinity_value is not None:
            labels['affinity'] = torch.tensor(float(affinity_value), dtype=torch.float32)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(labels, output_file)
        logging.info(
            f"Successfully processed and saved labels for {pdb_id} (Groups: {atom_group_ids.bincount().tolist()})")
        return 'success'

    except Exception as e:
        logging.error(f"Failed to process {pdb_id}: {e}", exc_info=True)
        return 'error'


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Pre-process complex files to generate labels with ground truth and initial coordinates.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    try:
        load_config(Path(args.config).resolve())
        setup_logging('phy_geo_log')
    except Exception as e:
        logging.critical(f"Failed to initialize script: {e}", exc_info=True)
        exit(1)

    logging.info("--- Starting Physics/Geometry Label Preprocessing ---")

    task_config = CONFIG.get('phy_geo_task', {})
    if not task_config:
        logging.critical("'phy_geo_task' not found in config.yaml. Aborting.")
        return

    affinity_files = task_config.get('affinity_files', [])
    affinity_map = load_affinity_data(affinity_files)

    data_sources = task_config.get('data_sources', [])
    if not data_sources:
        logging.warning("No 'data_sources' defined in the config. Nothing to process.")
        return

    total_results = defaultdict(int)

    for source in data_sources:
        source_name = source.get('name', 'UnnamedSource')
        source_type = source.get('type')
        input_path_str = source.get('input_dir')
        if not input_path_str:
            logging.error(f"Missing 'input_dir' for data source {source_name}. Skipping.")
            continue

        input_path = Path(input_path_str)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path

        logging.info(f"--- Processing ground truth data source: {source_name} (Type: {source_type}) ---")

        if source_type not in DATA_LOADERS:
            logging.error(f"Unknown data source type '{source_type}' for {source_name}. Skipping.")
            continue

        loader_func = DATA_LOADERS[source_type]

        items_to_process = []
        if source_type == "maegz_complex":
            pattern = source.get('file_pattern', '*/*_complex.maegz')
            items_to_process = list(input_path.glob(pattern))
        elif source_type == "pdb_sdf_pair":
            items_to_process = [d for d in input_path.iterdir() if d.is_dir()]

        if not items_to_process:
            logging.warning(f"No items found for data source {source_name} in {input_path}.")
            continue

        logging.info(f"Found {len(items_to_process)} items for {source_name}.")

        for item_path in tqdm(items_to_process, desc=f"Processing {source_name}"):
            ground_truth_st, pdb_id = loader_func(item_path)

            if ground_truth_st and pdb_id:
                status = process_entry(ground_truth_st, pdb_id, task_config, affinity_map)
                total_results[status] += 1

    logging.info("--- Preprocessing Complete ---")
    for status, count in total_results.items():
        logging.info(f"  {status.capitalize()}: {count}")


if __name__ == '__main__':
    main()