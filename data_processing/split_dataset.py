#!/usr/bin/env python
# Location: /home/zdy/Project2/data_processing/split_dataset.py
# A script to prepare and split a protein-ligand dataset.
# It performs multi-stage filtering, layered partitioning for multiple test sets,
# and sequence-based clustering for training/validation sets, enriching all
# outputs with affinity data.

import re
import logging
import subprocess
import random
import json
import datetime
from pathlib import Path
from typing import Set, List, Dict, Any

import yaml
import pandas as pd
from tqdm import tqdm

# --- Constants ---
# The project root is two levels up from the script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard amino acid 3-to-1 letter code mapping
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'TRP': 'W', 'THR': 'T',
    'TYR': 'Y', 'VAL': 'V'
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_cfg: Dict[str, Any]):
    """Configures logging for the script."""
    log_dir_rel = log_cfg.get('log_dir', 'logs/split_dataset')
    log_base_name = log_cfg.get('log_base_name', 'split_dataset')
    log_level_str = log_cfg.get('log_level', 'INFO').upper()
    log_dir_abs = PROJECT_ROOT / log_dir_rel
    log_dir_abs.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir_abs / f"{log_base_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.getLevelName(log_level_str),
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logging configured. Log file: {log_file_path}")


def get_pdb_ids_from_dir(dir_path: Path) -> Set[str]:
    """Scans a directory to get PDB IDs from subdirectory names (in lowercase)."""
    if not dir_path.is_dir():
        logging.warning(f"Directory not found: {dir_path}. Returning empty set.")
        return set()
    ids = {d.name.lower() for d in dir_path.iterdir() if d.is_dir() and len(d.name) == 4}
    logging.info(f"Found {len(ids)} potential PDB IDs in: {dir_path}")
    return ids


def parse_log_for_ids(log_file: Path, pattern_str: str, block_header: str = None) -> Set[str]:
    """A generic function to parse a log file for PDB IDs (in lowercase)."""
    if not log_file.is_file():
        logging.warning(f"Log file not found: {log_file}. No IDs extracted.")
        return set()
    ids = set()
    pattern = re.compile(pattern_str, re.IGNORECASE)
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    search_area = content
    if block_header:
        block_match = re.search(f"{re.escape(block_header)}(.*)", content, re.DOTALL)
        if block_match:
            search_area = block_match.group(1)
    matches = pattern.findall(search_area)
    ids.update(match.lower() for match in matches)
    logging.info(f"Found {len(ids)} IDs in {log_file.name} to exclude.")
    return ids


def get_ids_from_benchmark_file(file_path: Path) -> Set[str]:
    """Reads a benchmark file, extracts PDB IDs, and converts them to lowercase."""
    if not file_path.is_file():
        logging.warning(f"Benchmark file not found: {file_path}. No IDs extracted.")
        return set()
    ids = {line.strip()[:4].lower() for line in open(file_path, 'r', errors='ignore') if len(line.strip()) >= 4}
    logging.info(f"Found {len(ids)} IDs in benchmark file: {file_path.name}")
    return ids


def load_affinity_data(index_file_path: Path) -> Dict[str, float]:
    """Parses the raw PDBbind index file, ensuring PDB IDs are lowercase."""
    if not index_file_path.is_file():
        raise FileNotFoundError(f"PDBbind index file not found: {index_file_path}")
    affinity_map = {}
    with open(index_file_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    pdb_id = parts[0].lower()
                    affinity_value = float(parts[3])
                    affinity_map[pdb_id] = affinity_value
                except (ValueError, IndexError):
                    continue
    logging.info(f"Loaded affinity data for {len(affinity_map)} complexes.")
    return affinity_map


def extract_sequence_from_pdbqt(pdb_id: str, protein_dir: Path) -> str:
    """Extracts 1-letter amino acid sequence from a PDBQT file."""
    pdbqt_file = protein_dir / pdb_id / f"{pdb_id}_protein.pdbqt"
    if not pdbqt_file.is_file(): return ""
    sequence, last_res_id = [], None
    with open(pdbqt_file, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    res_id = (line[21], int(line[22:26]), line[26].strip())
                    if res_id != last_res_id:
                        one_letter_code = AA_3_TO_1.get(line[17:20].strip())
                        if one_letter_code: sequence.append(one_letter_code)
                        last_res_id = res_id
                except (ValueError, IndexError):
                    continue
    return "".join(sequence)


def run_cd_hit(fasta_path: Path, output_prefix: Path, identity: float, threads: int, cd_hit_exe_path: str) -> Path:
    """Runs CD-HIT to cluster sequences."""
    if not Path(cd_hit_exe_path).is_file():
        raise EnvironmentError(f"CD-HIT executable not found at specified path: {cd_hit_exe_path}")
    clstr_file = output_prefix.with_suffix(".clstr")
    command = [
        cd_hit_exe_path, "-i", str(fasta_path), "-o", str(output_prefix),
        "-c", str(identity), "-n", "5", "-d", "0", "-T", str(threads), "-M", "0"
    ]
    logging.info(f"Running CD-HIT with command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logging.error(f"CD-HIT execution failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise RuntimeError("CD-HIT failed.")
    logging.info("CD-HIT clustering completed successfully.")
    return clstr_file


def parse_cd_hit_clusters(clstr_file: Path) -> List[List[str]]:
    """Parses a .clstr file into a list of clusters."""
    clusters = []
    with open(clstr_file, 'r', errors='ignore') as f:
        current_cluster = []
        for line in f:
            if line.startswith(">Cluster"):
                if current_cluster: clusters.append(current_cluster)
                current_cluster = []
            else:
                match = re.search(r'>([a-zA-Z0-9]{4})\.\.\.', line)
                if match: current_cluster.append(match.group(1))
        if current_cluster: clusters.append(current_cluster)
    logging.info(f"Parsed {len(clusters)} clusters from CD-HIT output.")
    return clusters


def main():
    """Main execution function."""
    config_path = PROJECT_ROOT / 'config.yaml'
    try:
        config = load_config(config_path)
        setup_logging(config.get('logging', {}).get('split_dataset', {}))
    except Exception as e:
        logging.critical(f"FATAL: Could not initialize. Error: {e}")
        return

    try:
        cfg = config.get('dataset_split')
        if not cfg: raise ValueError("'dataset_split' section not found in config.yaml")

        # --- 1. Path Resolution ---
        logging.info("--- Step 1: Resolving Paths ---")
        output_dir = PROJECT_ROOT / cfg['output_dir'];
        output_dir.mkdir(parents=True, exist_ok=True)
        pdbbind_dir = PROJECT_ROOT / cfg['pdbbind_dir']
        pdbbind_index_file = PROJECT_ROOT / cfg['pdbbind_index_file']
        casf_dir = PROJECT_ROOT / cfg['casf_dir']
        protein_dir = PROJECT_ROOT / cfg['prepared_protein_dir']
        temp_dir = PROJECT_ROOT / cfg['temp_dir'];
        temp_dir.mkdir(parents=True, exist_ok=True)
        protein_prep_log = PROJECT_ROOT / cfg['protein_preparation_log']
        drug_prep_log = PROJECT_ROOT / cfg['drug_preparation_log']
        posebusters_ids_file = PROJECT_ROOT / cfg['posebusters_ids_file']
        astex_ids_file = PROJECT_ROOT / cfg['astex_ids_file']
        cd_hit_exe = cfg['cd_hit_executable_path']

        # --- 2. Initial Data Loading and Filtering ---
        logging.info("\n--- Step 2: Initial Data Loading and Filtering ---")
        all_pdbbind_ids = get_pdb_ids_from_dir(pdbbind_dir)
        affinity_map = load_affinity_data(pdbbind_index_file)

        master_df = pd.DataFrame(affinity_map.items(), columns=['pdb_id', 'affinity'])
        master_df = master_df[master_df['pdb_id'].isin(all_pdbbind_ids)].reset_index(drop=True)
        logging.info(f"Initial pool: {len(master_df)} IDs with both structure and affinity.")

        protein_failed_ids = parse_log_for_ids(protein_prep_log, r"ERROR -   ([a-zA-Z0-9]{4})_p\.pdb",
                                               "--- FAILED: prepare_receptor4.py errors")
        drug_failed_ids = parse_log_for_ids(drug_prep_log, r"WARNING - ([a-zA-Z0-9]{4})_l\.sdf",
                                            "--- SDFs with Charge Calculation Warnings")
        log_failed_ids = protein_failed_ids.union(drug_failed_ids)

        master_df = master_df[~master_df['pdb_id'].isin(log_failed_ids)].reset_index(drop=True)
        logging.info(f"After log filtering: {len(master_df)} IDs remain.")

        # --- 3. Layered Partitioning of Test Sets ---
        logging.info("\n--- Step 3: Partitioning Test Sets ---")

        remaining_df = master_df.copy()

        # Partition 1: CASF Test Set
        casf_ids = get_pdb_ids_from_dir(casf_dir)
        test_df = remaining_df[remaining_df['pdb_id'].isin(casf_ids)].copy()
        test_df.to_csv(output_dir / "test.csv", index=False)
        logging.info(f"Saved {len(test_df)} IDs to test.csv (from CASF intersection).")

        # Partition 2: PoseBusters Set (direct read from file)
        posebusters_ids = get_ids_from_benchmark_file(posebusters_ids_file)
        posebusters_df_data = [{'pdb_id': pid, 'affinity': affinity_map.get(pid, float('nan'))} for pid in
                               sorted(list(posebusters_ids))]
        posebusters_df = pd.DataFrame(posebusters_df_data)
        posebusters_df.to_csv(output_dir / "posebusters.csv", index=False)
        logging.info(f"Saved {len(posebusters_df)} IDs directly from PoseBusters list to posebusters.csv.")

        # Partition 3: Astex Set
        astex_ids = get_ids_from_benchmark_file(astex_ids_file)
        astex_df = remaining_df[remaining_df['pdb_id'].isin(astex_ids)].copy()
        astex_df.to_csv(output_dir / "astex.csv", index=False)
        logging.info(f"Saved {len(astex_df)} IDs to astex.csv (from Astex intersection).")

        # Define the final exclusion set from the master pool
        exclusion_ids = set(test_df['pdb_id']).union(posebusters_ids).union(set(astex_df['pdb_id']))

        train_val_df = master_df[~master_df['pdb_id'].isin(exclusion_ids)].reset_index(drop=True)
        logging.info(f"After all test set partitions, {len(train_val_df)} IDs remain for training/validation.")

        # --- 4. Sequence Clustering for Train/Val Split ---
        logging.info("\n--- Step 4: Sequence Clustering for Train/Val Split ---")
        fasta_file = temp_dir / "train_val_pool.fasta"

        with open(fasta_file, 'w') as f:
            for pdb_id in tqdm(train_val_df['pdb_id'], desc="Generating FASTA file"):
                sequence = extract_sequence_from_pdbqt(pdb_id, protein_dir)
                if sequence:
                    f.write(f">{pdb_id}\n{sequence}\n")

        if not fasta_file.exists() or fasta_file.stat().st_size == 0:
            logging.warning("FASTA file for clustering is empty. Skipping clustering and performing random split.")
            clusters = [[pid] for pid in train_val_df['pdb_id']]
        else:
            try:
                cluster_file = run_cd_hit(
                    fasta_file,
                    temp_dir / "cd_hit_output",
                    cfg['cd_hit_identity'],
                    cfg.get('cd_hit_threads', 8),
                    cd_hit_exe
                )
                clusters = parse_cd_hit_clusters(cluster_file)
            except (EnvironmentError, RuntimeError) as e:
                logging.error(f"CD-HIT failed: {e}. Falling back to random split.")
                clusters = [[pid] for pid in train_val_df['pdb_id']]

        # --- 5. Final Splitting and Saving ---
        logging.info("\n--- Step 5: Final Splitting and Saving ---")
        random.seed(cfg['random_seed'])
        random.shuffle(clusters)

        split_idx = int(len(clusters) * (1.0 - cfg['validation_set_ratio']))
        train_clusters, valid_clusters = clusters[:split_idx], clusters[split_idx:]
        logging.info(
            f"Splitting {len(clusters)} clusters -> Train: {len(train_clusters)}, Valid: {len(valid_clusters)}")

        train_ids = {pid for cl in train_clusters for pid in cl}
        valid_ids = {pid for cl in valid_clusters for pid in cl}

        final_train_df = train_val_df[train_val_df['pdb_id'].isin(train_ids)]
        final_valid_df = train_val_df[train_val_df['pdb_id'].isin(valid_ids)]

        final_train_df.to_csv(output_dir / "train.csv", index=False)
        logging.info(f"Saved {len(final_train_df)} training entries to data/train.csv")

        final_valid_df.to_csv(output_dir / "valid.csv", index=False)
        logging.info(f"Saved {len(final_valid_df)} validation entries to data/valid.csv")

        # Save the detailed cluster assignment map
        cluster_map_path = output_dir / "cluster_split_details.json"

        cluster_data_to_save = {
            "train_clusters": train_clusters,
            "valid_clusters": valid_clusters
        }

        with open(cluster_map_path, 'w') as f:
            json.dump(cluster_data_to_save, f, indent=2)

        logging.info(f"Saved detailed cluster assignments to {cluster_map_path}")

        logging.info("--- Dataset Splitting Process Finished Successfully ---")

    except Exception as e:
        logging.critical(f"An unexpected error occurred during the process: {e}", exc_info=True)

if __name__ == "__main__":
    main()