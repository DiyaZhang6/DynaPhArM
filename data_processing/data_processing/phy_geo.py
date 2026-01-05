#!/usr/bin/env python
# Pre-processes high-quality complex files to extract ground truth labels using RDKit and BioPython.

import argparse
import logging
import yaml
import datetime
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Tuple, Optional, Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Third-party imports ---
try:
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs
    from rdkit.Chem import AllChem
    from Bio.PDB import PDBParser, Structure
    from rdkit.Chem import rdForceFieldHelpers as FFHelpers
    import pandas as pd
except ImportError as e:
    print(f"FATAL: This script requires RDKit, BioPython, and pandas. Error: {e}")
    exit(1)


# --- Global Config ---
CONFIG = None
PROJECT_ROOT = None
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
_STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}


# --- Configuration & Logging ---
def setup_logging(log_config_key: str, project_root: Path, config: Dict):
    log_config = config.get('logging', {}).get(log_config_key, {})
    log_level = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{log_base_name}_{timestamp}.log" if use_timestamp else f"{log_base_name}.log"
    log_path = project_root / log_dir_rel / log_file_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logging for '{log_config_key}' configured. Level: {log_level}. Output file: {log_path}")


def load_config(config_path: Path) -> Tuple[Dict, Path]:
    if not config_path.is_file(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.resolve()
    return cfg, project_root

# --- Data Loading Abstraction ---
def load_affinity_data(file_paths: List[str], project_root: Path) -> Dict[str, float]:
    affinity_map = {}
    if not file_paths:
        logging.warning("No affinity files specified in config. No affinity labels will be generated.")
        return affinity_map
    logging.info("Loading affinity data from specified files...")
    for file_path in file_paths:
        path = project_root / file_path
        if not path.exists():
            logging.warning(f"Affinity data file not found: {path}. Skipping.")
            continue
        try:
            df = pd.read_csv(path)
            df.dropna(subset=['pdb_id', 'affinity'], inplace=True)
            for _, row in df.iterrows():
                affinity_map[str(row['pdb_id']).lower()] = float(row['affinity'])
            logging.info(f"Loaded {len(df)} valid entries from {path.name}. Total unique entries: {len(affinity_map)}")
        except Exception as e:
            logging.error(f"Failed to read or process affinity data from {path}: {e}")
    return affinity_map


def load_from_pdb_sdf_pair(dir_path: Path) -> Tuple[Optional[Chem.Mol], Optional[str], Optional[Structure.Structure]]:
    pdb_id = dir_path.name.lower()
    protein_path = dir_path / f"{pdb_id}_protein.pdb"
    ligand_path = dir_path / f"{pdb_id}_ligand.sdf"

    if not all(p.exists() for p in [protein_path, ligand_path]):
        return None, None, None
    try:
        ligand_mol_supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=False)
        ligand_mol = next(ligand_mol_supplier, None) if ligand_mol_supplier else None

        protein_mol = Chem.MolFromPDBFile(str(protein_path), removeHs=False, sanitize=False)

        if not protein_mol:
            logging.warning(f"[{pdb_id}] Failed to load protein from: {protein_path}")
            return None, None, None
        if not ligand_mol:
            logging.warning(f"[{pdb_id}] Failed to load ligand from: {ligand_path}")
            return None, None, None

        protein_mol = Chem.AddHs(protein_mol, addCoords=True)
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)

        parser = PDBParser(QUIET=True)
        protein_struct_bp = parser.get_structure(pdb_id, str(protein_path))

        complex_mol = Chem.CombineMols(protein_mol, ligand_mol)
        return complex_mol, pdb_id, protein_struct_bp
    except Exception as e:
        logging.warning(f"Failed to load PDB/SDF pair for {pdb_id}: {e}")
        return None, None, None


def load_initial_complex(pdb_id: str, task_config: dict, project_root: Path) -> Optional[Chem.Mol]:
    init_struct_dir = project_root / task_config.get('initial_structure_dir')
    dir_path = init_struct_dir / pdb_id
    if not dir_path.is_dir(): return None

    templates = task_config.get('split_file_templates', {})
    backbone_path = dir_path / f"{pdb_id}{templates.get('backbone_suffix', '_backbone.pdb')}"
    sidechain_path = dir_path / f"{pdb_id}{templates.get('sidechain_suffix', '_sidechain.pdb')}"
    ligand_path = dir_path / f"{pdb_id}{templates.get('ligand_suffix', '_ligand.sdf')}"

    try:
        if not all(p.exists() for p in [backbone_path, sidechain_path, ligand_path]): return None

        backbone_mol = Chem.MolFromPDBFile(str(backbone_path), removeHs=False, sanitize=False)
        sidechain_mol = Chem.MolFromPDBFile(str(sidechain_path), removeHs=False, sanitize=False)
        ligand_mol = next(Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=False), None)

        if not all([backbone_mol, sidechain_mol, ligand_mol]): return None

        backbone_mol = Chem.AddHs(backbone_mol, addCoords=True)
        sidechain_mol = Chem.AddHs(sidechain_mol, addCoords=True)
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)

        init_complex = Chem.CombineMols(Chem.CombineMols(backbone_mol, sidechain_mol), ligand_mol)
        return init_complex
    except Exception:
        return None


# --- Feature Extraction Functions ---
def get_atom_groups_and_charges(complex_mol: Chem.Mol, protein_struct_bp: Structure.Structure) -> Tuple[
    torch.Tensor, torch.Tensor]:
    AllChem.ComputeGasteigerCharges(complex_mol)
    partial_charges = [float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0 for atom in
                       complex_mol.GetAtoms()]

    protein_atoms_bp = list(protein_struct_bp.get_atoms())
    num_protein_atoms = len(protein_atoms_bp)

    group_ids = []
    # Use BioPython structure for reliable protein atom grouping
    for atom_bp in protein_atoms_bp:
        resname = atom_bp.get_parent().get_resname()
        atom_name = atom_bp.get_name().strip()

        if resname in _STANDARD_AMINO_ACIDS:
            if atom_name in _BACKBONE_ATOMS:
                group_ids.append(0)
            else:
                group_ids.append(1)
        else:
            group_ids.append(1)  # Classify non-standard residues/HETATMs in protein as sidechain

    # Ligand atoms are the remaining atoms in the combined mol
    num_ligand_atoms = complex_mol.GetNumAtoms() - num_protein_atoms
    group_ids.extend([2] * num_ligand_atoms)

    # Final check for consistency
    if len(group_ids) != complex_mol.GetNumAtoms():
        logging.warning(
            "Atom count mismatch between BioPython and RDKit protein representations. Grouping may be inaccurate.")
        # Failsafe: Re-calculate based on RDKit's monomer info if mismatch
        group_ids = []
        for atom in complex_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info and info.GetResidueName() in _STANDARD_AMINO_ACIDS:
                if info.GetName().strip() in _BACKBONE_ATOMS:
                    group_ids.append(0)
                else:
                    group_ids.append(1)
            else:
                group_ids.append(2)

    return torch.tensor(group_ids, dtype=torch.long), torch.tensor(partial_charges, dtype=torch.float32)


def get_vdw_radii(mol, config: dict):
    vdw_radii_map = config['vdw_radii']
    default_radius = config['default_vdw_radius']
    radii = [vdw_radii_map.get(atom.GetSymbol(), default_radius) for atom in mol.GetAtoms()]
    return torch.tensor(radii, dtype=torch.float32)


def get_topology_and_references(mol: Chem.Mol, config: dict, pdb_id: str) -> Optional[Dict]:
    """
    Extracts topology and reference values. This is a robust version that handles
    RDKit API inconsistencies and data quality issues.
    """
    # --- Step 1: Sanitize the molecule ---
    try:
        mol_copy = Chem.Mol(mol)
        blocker = BlockLogs()
        sanitize_status = Chem.SanitizeMol(mol_copy, catchErrors=True)
        del blocker
        if sanitize_status != Chem.SanitizeFlags.SANITIZE_NONE:
            logging.warning(
                f"[{pdb_id}] Sanitization failed with status {sanitize_status}. Skipping topology extraction.")
            return None
    except Exception as e:
        logging.warning(f"[{pdb_id}] Sanitization failed with a severe error: {e}. Skipping topology extraction.")
        return None

    # --- Step 2: Determine which forcefield is available ---
    ff_props = None
    ff_type = 'None'
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_copy):
            ff_props = AllChem.MMFFGetMoleculeProperties(mol_copy)
            if AllChem.MMFFGetMoleculeForceField(mol_copy, ff_props):
                ff_type = 'MMFF94'
        elif AllChem.UFFHasAllMoleculeParams(mol_copy):
            ff_type = 'UFF'
    except Exception as e:
        logging.debug(f"[{pdb_id}] Force field check failed: {e}. Using defaults.")
        ff_type = 'None'

    # --- Step 3: Extract topology ---
    bonds_list, angles_list, dihedrals_list = [], [], []
    ref_bonds, ref_angles, true_dihedrals_rad = [], [], []
    default_bond_len = config['default_bond_length']
    default_angle_rad = np.deg2rad(config['default_bond_angle_deg'])

    # Process Bonds
    for bond in mol_copy.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds_list.append(sorted((i, j)))

        params = None
        try:
            if ff_type == 'MMFF94':
                params = None
                logging.debug(f"[{pdb_id}] MMFF detected. Using default bond length for bond ({i}-{j}).")
            elif ff_type == 'UFF':
                params = FFHelpers.GetUFFBondStretchParams(mol_copy, i, j)

        except RuntimeError as e:
            logging.warning(f"[{pdb_id}] RDKit RuntimeError getting bond params for ({i}-{j}): {e}. Using default.")
            params = None

        # UFF params are (r0, kb).
        ref_bonds.append(params[0] if params else default_bond_len)

    # Process Angles
    for i in range(mol_copy.GetNumAtoms()):
        atom = mol_copy.GetAtomWithIdx(i)
        if atom.GetDegree() >= 2:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for j_idx in range(len(neighbors)):
                for k_idx in range(j_idx + 1, len(neighbors)):
                    j, k = neighbors[j_idx], neighbors[k_idx]
                    angles_list.append(sorted((j, i, k)))

                    params = None
                    try:
                        if ff_type == 'MMFF94':
                            params = None
                            logging.debug(f"[{pdb_id}] MMFF detected. Using default angle for ({j}-{i}-{k}).")
                        elif ff_type == 'UFF':
                            params = FFHelpers.GetUFFAngleBendParams(mol_copy, j, i, k)

                    except RuntimeError as e:
                        logging.warning(
                            f"[{pdb_id}] RDKit RuntimeError getting angle params for ({j}-{i}-{k}): {e}. Using default.")
                        params = None

                    # UFF params are (theta0, ka).
                    angle_deg = params[0] if params else None
                    ref_angles.append(np.deg2rad(angle_deg) if angle_deg is not None else default_angle_rad)

    # Process Dihedrals
    conformer = mol.GetConformer()
    if conformer:
        for bond in mol_copy.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            atom_i, atom_j = mol_copy.GetAtomWithIdx(i), mol_copy.GetAtomWithIdx(j)
            neighbors_i = [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j]
            neighbors_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i]
            for h in neighbors_i:
                for k in neighbors_j:
                    if h != k:
                        dihedral_indices = (h, i, j, k)
                        dihedrals_list.append(dihedral_indices)
                        angle_rad = AllChem.GetDihedralRad(conformer, h, i, j, k)
                        true_dihedrals_rad.append(angle_rad)

    # --- Step 4: Convert to Tensors and Pre-compute Sets ---
    bond_set = {tuple(p) for p in bonds_list}
    angle_1_3_set = {tuple(sorted((p[0], p[2]))) for p in angles_list}

    bonds_tensor = torch.tensor(bonds_list, dtype=torch.long) if bonds_list else torch.empty((0, 2), dtype=torch.long)
    angles_tensor = torch.tensor(angles_list, dtype=torch.long) if angles_list else torch.empty((0, 3),
                                                                                                dtype=torch.long)
    dihedrals_tensor = torch.tensor(dihedrals_list, dtype=torch.long) if dihedrals_list else torch.empty((0, 4),
                                                                                                         dtype=torch.long)

    # --- Step 5: Return final dictionary ---
    return {
        'bonds': bonds_tensor,
        'ref_lengths': torch.tensor(ref_bonds, dtype=torch.float32),
        'angles': angles_tensor,
        'ref_angles': torch.tensor(ref_angles, dtype=torch.float32),
        'dihedrals': dihedrals_tensor,
        'true_angles': torch.tensor(true_dihedrals_rad, dtype=torch.float32),
        'bond_set': bond_set,
        'angle_1_3_set': angle_1_3_set
    }

def get_ring_info(mol, config: dict):
    aromatic_rings = []
    positions = mol.GetConformer().GetPositions()
    norm_thresh = config['ring_normal_norm_threshold']
    try:
        Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        Chem.AssignStereochemistryFrom3D(mol)
        Chem.SetAromaticity(mol)
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
    except Exception as e:
        logging.debug(f"Ring info calculation failed: {e}")
    return aromatic_rings


def get_interaction_labels(mol, topology, config: dict):
    num_atoms = mol.GetNumAtoms()
    all_pairs = torch.combinations(torch.arange(num_atoms), r=2)
    bonded_set = {tuple(sorted(p)) for p in topology['bonds'].tolist()}
    angle_set = {tuple(sorted((p[0], p[2]))) for p in topology['angles'].tolist()}
    non_bonded_mask = torch.tensor(
        [tuple(sorted(p.tolist())) not in bonded_set and tuple(sorted(p.tolist())) not in angle_set for p in all_pairs])

    hbond_triplets = []
    try:
        donor_s = Chem.MolFromSmarts(config['hbond_donors_smarts'])
        acceptor_s = Chem.MolFromSmarts(config['hbond_acceptors_smarts'])
        donors_idx = sum(mol.GetSubstructMatches(donor_s), ())
        acceptors_idx = sum(mol.GetSubstructMatches(acceptor_s), ())
        for d_idx in donors_idx:
            for h_atom in mol.GetAtomWithIdx(d_idx).GetNeighbors():
                if h_atom.GetAtomicNum() == 1:
                    for a_idx in acceptors_idx:
                        if d_idx != a_idx:
                            hbond_triplets.append((d_idx, h_atom.GetIdx(), a_idx))
    except Exception as e:
        logging.debug(f"HBond search failed: {e}")

    rings = get_ring_info(mol, config)
    pi_pi_ring_pair_indices = []
    pi_pi_dist_cutoff = config['pi_pi_distance_cutoff']
    if len(rings) >= 2:
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                d_ij = np.linalg.norm(rings[i]['centroid'] - rings[j]['centroid'])
                if d_ij <= pi_pi_dist_cutoff:
                    pi_pi_ring_pair_indices.append(
                        [torch.tensor(rings[i]['indices'], dtype=torch.long),
                         torch.tensor(rings[j]['indices'], dtype=torch.long)])
    return {
        'non_bonded_pairs': all_pairs[non_bonded_mask],
        'hbond_triplets': torch.tensor(hbond_triplets, dtype=torch.long) if hbond_triplets else torch.empty((0, 3),
                                                                                                            dtype=torch.long),
        'pi_pi_ring_pair_indices': pi_pi_ring_pair_indices,
    }


# --- Main Processing Logic ---
def get_atom_id(atom: Chem.Atom):
    info = atom.GetPDBResidueInfo()
    if info:
        return f"{info.GetChainId()}_{info.GetResidueNumber()}_{info.GetResidueName().strip()}_{info.GetName().strip()}"
    return f"UNK_0_UNK_{atom.GetIdx()}"


def generate_labels_for_pdb_id(pdb_id: str, ground_truth_mol: Chem.Mol, protein_struct_bp: Structure.Structure,
                               task_config: dict, affinity_map: dict, project_root: Path) -> Optional[dict]:
    """
    Core logic to generate all physical and geometric labels for a given PDB entry.

    This function takes the loaded ground truth molecules and performs all necessary calculations
    to produce the final labels dictionary. It returns the dictionary on success or None on critical failure.
    """
    try:
        # --- Step 1: Extract Ground Truth and Initial Coordinates ---

        # Get ground truth coordinates and atom identifiers
        r_true = torch.from_numpy(ground_truth_mol.GetConformer().GetPositions().astype(np.float32))
        true_atom_ids = [get_atom_id(atom) for atom in ground_truth_mol.GetAtoms()]

        # Load the initial complex structure
        init_complex_mol = load_initial_complex(pdb_id, task_config, project_root)
        r_init = None

        if init_complex_mol:
            # Check for atom count mismatch between initial and ground truth structures
            if init_complex_mol.GetNumAtoms() != ground_truth_mol.GetNumAtoms():
                logging.warning(
                    f"[{pdb_id}] Atom count mismatch! True: {ground_truth_mol.GetNumAtoms()}, "
                    f"Init: {init_complex_mol.GetNumAtoms()}. Attempting atom-by-atom alignment."
                )
                # Create a map from unique atom ID string to index for the initial structure
                init_atom_map = {get_atom_id(atom): atom.GetIdx() for atom in init_complex_mol.GetAtoms()}
                init_coords_all = init_complex_mol.GetConformer().GetPositions()
                aligned_coords, found_ids = [], set()

                # Iterate through ground truth atoms and find corresponding coordinates in the initial structure
                for i, true_id in enumerate(true_atom_ids):
                    if true_id in init_atom_map and true_id not in found_ids:
                        aligned_coords.append(init_coords_all[init_atom_map[true_id]])
                        found_ids.add(true_id)
                    else:
                        # If an atom is not found, use its ground truth coordinate as a placeholder
                        aligned_coords.append(r_true[i].numpy())

                r_init = torch.from_numpy(np.array(aligned_coords, dtype=np.float32))
                logging.info(f"[{pdb_id}] Successfully aligned {len(found_ids)}/{len(true_atom_ids)} atoms.")
            else:
                # Atom counts match, directly use the coordinates
                r_init = torch.from_numpy(init_complex_mol.GetConformer().GetPositions().astype(np.float32))

        # Fallback: If initial structure failed to load, create r_init by adding noise to r_true
        if r_init is None:
            logging.info(f"[{pdb_id}] No initial structure found or loaded. Using noise fallback for r_init.")
            noise = torch.randn_like(r_true) * task_config.get('fallback_noise_scale', 1.0)
            r_init = r_true + noise

        # --- Step 2: Extract Topology and Basic Atomic Properties ---

        # Extract covalent topology (bonds, angles, dihedrals) and reference values from force fields
        topology = get_topology_and_references(ground_truth_mol, task_config, pdb_id)
        if topology is None:
            logging.warning(f"[{pdb_id}] Skipping due to severe topology extraction errors.")
            return None

        # Get atom groups (backbone=0, sidechain=1, ligand=2) and partial charges
        atom_group_ids, partial_charges = get_atom_groups_and_charges(ground_truth_mol, protein_struct_bp)

        # Get atomic numbers
        atomic_nums = torch.tensor([atom.GetAtomicNum() for atom in ground_truth_mol.GetAtoms()], dtype=torch.long)

        # Get Van der Waals radii
        vdw_radii = get_vdw_radii(ground_truth_mol, task_config)

        # --- Step 3: Extract Interaction Labels ---

        # Identify non-bonded pairs, hydrogen bond triplets, and pi-pi stacking ring pairs
        interactions = get_interaction_labels(ground_truth_mol, topology, task_config)

        # --- Step 4: Extract Affinity Data ---

        # Look up the binding affinity value from the pre-loaded map
        affinity_value = affinity_map.get(pdb_id.lower())

        # --- Step 5: Assemble the Final Labels Dictionary ---

        labels = {
            'r_true': r_true,
            'r_init': r_init,
            'atomic_nums': atomic_nums,
            'atom_group_ids': atom_group_ids,
            'partial_charges': partial_charges,
            'bond': {
                'indices': topology['bonds'],
                'ref_lengths': topology['ref_lengths']
            },
            'angle': {
                'indices': topology['angles'],
                'ref_angles': topology['ref_angles']
            },
            'dihedral': {
                'indices': topology['dihedrals'],
                'true_angles': topology['true_angles']
            },
            'vdw': {
                'indices': interactions['non_bonded_pairs'],
                'radii': vdw_radii
            },
            'hbond': {
                'indices': interactions['hbond_triplets']
            },
            'pi_pi': {
                'ring_pair_indices': interactions['pi_pi_ring_pair_indices']
            },
            'electro': {
                'indices': interactions['non_bonded_pairs']
            }
        }

        # Add affinity to the labels dictionary only if it was found
        if affinity_value is not None:
            labels['affinity'] = torch.tensor(float(affinity_value), dtype=torch.float32)

        return labels

    except Exception as e:
        # Catch any unexpected errors during the process
        logging.error(f"Core logic failed for {pdb_id}", exc_info=True)
        return None

# --- Worker Function for Parallel Processing ---
def worker_process_entry(item_path: Path, task_config: dict, affinity_map: dict, project_root: Path) -> str:
    """
    Worker function wrapper for parallel processing.
    Handles loading, saving, and calls the core label generation logic.
    """
    try:
        # Step 1: Load ground truth data
        ground_truth_mol, pdb_id, protein_struct_bp = load_from_pdb_sdf_pair(item_path)
        if not (ground_truth_mol and pdb_id and protein_struct_bp):
            return 'load_error'

        # Step 2: Define output path and check if skipping is needed
        output_dir = project_root / task_config['output_dir']
        output_file = output_dir / pdb_id / f"{pdb_id}_labels.pt"
        if not task_config.get('overwrite_existing', False) and output_file.exists():
            return 'skipped'

        # Step 3: Call the core logic function to generate labels
        labels = generate_labels_for_pdb_id(
            pdb_id, ground_truth_mol, protein_struct_bp,
            task_config, affinity_map, project_root
        )

        # Step 4: Check result and save to file
        if labels is None:
            return 'processing_error'  # A specific status for failure in the core logic

        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(labels, output_file)
        return 'success'

    except Exception:
        logging.error(f"Worker encountered a fatal error for {item_path.name}", exc_info=True)
        return 'fatal_error'

def main():
    parser = argparse.ArgumentParser(description="Pre-process complex files in parallel to generate labels.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path

    try:
        config, project_root = load_config(config_path)
        setup_logging('phy_geo_log', project_root, config)
    except Exception as e:
        # Use basic print since logging might not be set up
        print(f"FATAL: Could not initialize script: {e}")
        exit(1)

    task_config = config.get('pipeline_tasks', {}).get('phy_geo_task', {})
    affinity_map = load_affinity_data(task_config.get('affinity_files', []), project_root)
    data_sources = task_config.get('data_sources', [])

    # --- Collect all items to process from all data sources ---
    all_items_to_process = []
    for source in data_sources:
        source_type = source.get('type')
        if source_type != "pdb_sdf_pair":
            logging.warning(f"Unsupported source type '{source_type}'. Skipping.")
            continue

        input_dir = project_root / source.get('input_dir')
        items = [d for d in input_dir.iterdir() if d.is_dir()]
        all_items_to_process.extend(items)
        logging.info(f"Found {len(items)} items for source: {source.get('name')}.")

    logging.info(f"\nTotal items to process across all sources: {len(all_items_to_process)}")

    # --- Setup and run multiprocessing pool ---
    mp_config = config.get('multiprocessing', {})
    num_workers = mp_config.get('num_workers')
    if not num_workers or num_workers == 0:
        num_workers = cpu_count()
    logging.info(f"Starting parallel processing with {num_workers} worker processes.")

    # Use functools.partial to create a worker function with fixed arguments
    worker_func = partial(worker_process_entry,
                          task_config=task_config,
                          affinity_map=affinity_map,
                          project_root=project_root)

    results = []
    with Pool(processes=num_workers) as pool:
        # imap_unordered is memory efficient and provides results as they complete
        # tqdm will wrap the iterator to show progress
        pbar = tqdm(pool.imap_unordered(worker_func, all_items_to_process),
                    total=len(all_items_to_process),
                    desc="Processing Complexes")
        for result in pbar:
            results.append(result)

    # --- Tally and report results ---
    logging.info("\n--- Preprocessing Complete ---")
    result_counts = Counter(results)
    for status, count in result_counts.items():
        logging.info(f"  {status.capitalize()}: {count}")


if __name__ == '__main__':
    main()