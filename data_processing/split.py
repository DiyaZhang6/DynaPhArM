#!/usr/bin/env python
# /home/zdy/Project2/data_processing/split.py
# Reads original protein files to split into backbone and sidechain.
# Reads docked PDBQT complex files to extract the best ligand pose.
# Writes all three components to a structured output directory.

import os
import logging
import logging.handlers
import warnings
import yaml
import multiprocessing
import queue
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TypedDict

# --- Third-party imports ---
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from openbabel import pybel, openbabel

# --- Global Variables (for worker processes) ---
CONFIG: Optional[Dict] = None
PROJECT_ROOT: Optional[Path] = None
log_queue: Optional[multiprocessing.Queue] = None

# --- Constants ---
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# --- Type Definitions for Clarity ---
class ProteinComponents(TypedDict):
    backbone_atoms: List[Atom]
    sidechain_atoms: List[Atom]


class WorkerResult(TypedDict):
    status: str  # 'success', 'no_protein', 'no_complex', 'split_error', 'save_error'
    pdb_id: str
    message: Optional[str]


# --- Logging Setup ---
def setup_main_logging_handlers(log_config_key: str) -> None:
    if not CONFIG or not PROJECT_ROOT:
        print("FATAL: CONFIG or PROJECT_ROOT not loaded.")
        return

    log_config = CONFIG.get('logging', {}).get(log_config_key, {})
    if not log_config:
        print(f"FATAL: Logging configuration for '{log_config_key}' not found.")
        return

    log_level_str = log_config.get('log_level', 'INFO').upper()
    log_dir_rel = log_config.get('log_dir', f'logs/{log_config_key}')
    log_base_name = log_config.get('log_base_name', log_config_key)
    use_timestamp = log_config.get('use_timestamp_in_log_name', False)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{log_base_name}_{timestamp}.log"
    else:
        log_file_name = f"{log_base_name}.log"

    log_path_abs = PROJECT_ROOT / log_dir_rel / log_file_name

    logger = logging.getLogger()
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            if not isinstance(handler, logging.handlers.QueueHandler):
                logger.removeHandler(handler)
    logger.setLevel(logging.getLevelName(log_level_str))

    log_path_abs.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path_abs, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.getLevelName(log_level_str))
    logger.addHandler(stream_handler)

    logging.info(f"Logging for '{log_config_key}' configured. Level: {log_level_str}. File: {log_path_abs}")


def setup_worker_logging(q: multiprocessing.Queue) -> None:
    openbabel.obErrorLog.SetOutputLevel(0)
    queue_handler = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.handlers.QueueHandler) for h in root_logger.handlers):
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.DEBUG)


# --- Initialization and Configuration ---
def initialize_worker(q: multiprocessing.Queue, cfg: dict, pr_root: Path):
    global log_queue, CONFIG, PROJECT_ROOT
    log_queue = q
    CONFIG = cfg
    PROJECT_ROOT = pr_root
    setup_worker_logging(log_queue)


def load_config(config_path: Path) -> None:
    """Loads the YAML config and sets the global PROJECT_ROOT."""
    global CONFIG, PROJECT_ROOT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    if 'project_base_dir' in CONFIG and CONFIG['project_base_dir'] != '.':
        # If project_base_dir is an absolute path or a different relative path
        PROJECT_ROOT = Path(CONFIG['project_base_dir']).resolve()
    else:
        # project root is the directory containing the config file.
        PROJECT_ROOT = config_path.parent.resolve()

    logging.info(f"Project root set to: {PROJECT_ROOT}")

# --- Core Logic Functions ---
def get_protein_components(protein_pdb_path: Path) -> Optional[ProteinComponents]:
    """Parses a standard PDB file to get backbone and sidechain atoms."""
    parser = PDBParser(QUIET=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = parser.get_structure(protein_pdb_path.stem, str(protein_pdb_path))
    except Exception as e:
        logging.error(f"Bio.PDB failed to parse protein {protein_pdb_path.name}: {e}")
        return None

    backbone, sidechain = [], []
    for atom in structure.get_atoms():
        if atom.get_parent().id[0] == ' ':  # It's a standard residue
            if atom.get_name().strip().upper() in _BACKBONE_ATOMS:
                backbone.append(atom)
            else:
                sidechain.append(atom)

    final_backbone = [a for a in backbone if getattr(a, 'element', 'H').upper() != 'H']
    final_sidechain = [a for a in sidechain if getattr(a, 'element', 'H').upper() != 'H']

    logging.info(f"Split protein {protein_pdb_path.name}: "
                 f"{len(final_backbone)} backbone, {len(final_sidechain)} sidechain atoms.")
    return {'backbone_atoms': final_backbone, 'sidechain_atoms': final_sidechain}


def save_ligand_from_complex(complex_pdbqt_path: Path, output_sdf_path: Path) -> bool:
    """Extracts the first model's ligand from a Vina PDBQT file and saves it as SDF."""
    try:
        # Using a molecule iterator to only read the first model
        mol_generator = pybel.readfile(format='pdbqt', filename=str(complex_pdbqt_path))

        # The first molecule read is the receptor, the second is the best ligand pose
        receptor = next(mol_generator)
        ligand_mol = next(mol_generator)

        ligand_mol.write("sdf", str(output_sdf_path), overwrite=True)
        logging.info(
            f"Successfully wrote {len(ligand_mol.atoms)} ligand atoms from best pose to {output_sdf_path.name}")
        return True
    except StopIteration:
        logging.error(f"File {complex_pdbqt_path.name} does not contain at least two molecules (receptor and ligand).")
        return False
    except Exception as e:
        logging.error(f"Failed to save ligand from {complex_pdbqt_path.name}: {e}", exc_info=True)
        return False


def save_atoms_as_pdb(atoms: List[Atom], output_path: Path) -> None:
    """Saves a list of Bio.PDB.Atom objects to a PDB file."""
    with open(output_path, 'w') as f:
        for atom in atoms:
            record_type = "HETATM" if atom.get_parent().id[0].strip() != '' else "ATOM"
            line = (f"{record_type:<6}{atom.get_serial_number():>5} {atom.get_name():<4}"
                    f"{atom.get_altloc():1}{atom.get_parent().get_resname():<3} "
                    f"{atom.get_parent().get_parent().get_id():1}{atom.get_parent().id[1]:>4}"
                    f"{atom.get_parent().id[2]:1}   {atom.get_coord()[0]:8.3f}"
                    f"{atom.get_coord()[1]:8.3f}{atom.get_coord()[2]:8.3f}"
                    f"{atom.get_occupancy():6.2f}{atom.get_bfactor():6.2f}          "
                    f"{getattr(atom, 'element', '  '):>2}")
            f.write(line + '\n')
    logging.debug(f"  Successfully wrote {len(atoms)} atoms to {output_path.name}")


# --- Main Worker Function ---
def process_pdb_id(pdb_id: str) -> WorkerResult:
    """The main worker function that handles splitting for a single PDB ID."""
    task_config = CONFIG['split_task']

    # --- Step 1: Find and process the original protein file ---
    protein_source_config = task_config['protein_source']
    prot_base = PROJECT_ROOT / protein_source_config['base_path']
    prot_subfolder = protein_source_config['subfolder_template'].format(pdb_id=pdb_id)
    prot_filename = protein_source_config['file_template'].format(pdb_id=pdb_id)
    protein_path = prot_base / prot_subfolder / prot_filename

    if not protein_path.is_file():
        msg = f"Original protein file not found for {pdb_id} at {protein_path}"
        logging.warning(msg)
        return WorkerResult(status='no_protein', pdb_id=pdb_id, message=msg)

    protein_components = get_protein_components(protein_path)
    if not protein_components or not protein_components['backbone_atoms']:
        msg = f"Failed to split protein components for {pdb_id} from {protein_path.name}"
        logging.error(msg)
        return WorkerResult(status='split_error', pdb_id=pdb_id, message=msg)

    # --- Step 2: Find and process the docked complex file for the ligand ---
    ligand_source_config = task_config['ligand_source']
    lig_base = PROJECT_ROOT / ligand_source_config['base_path']
    lig_subfolder = ligand_source_config['subfolder_template'].format(pdb_id=pdb_id)
    lig_filename = ligand_source_config['file_template'].format(pdb_id=pdb_id)
    complex_path = lig_base / lig_subfolder / lig_filename

    if not complex_path.is_file():
        msg = f"Docked complex file not found for {pdb_id} at {complex_path}"
        logging.warning(msg)
        return WorkerResult(status='no_complex', pdb_id=pdb_id, message=msg)

    # --- Step 3: Create output directory and save all components ---
    output_dir = PROJECT_ROOT / task_config['output_dir'] / pdb_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save protein components
        save_atoms_as_pdb(
            protein_components['backbone_atoms'],
            output_dir / f"{pdb_id}{task_config['backbone_suffix']}"
        )
        save_atoms_as_pdb(
            protein_components['sidechain_atoms'],
            output_dir / f"{pdb_id}{task_config['sidechain_suffix']}"
        )

        # Save ligand from complex
        ligand_ok = save_ligand_from_complex(
            complex_path,
            output_dir / f"{pdb_id}{task_config['ligand_suffix']}"
        )

        if not ligand_ok:
            return WorkerResult(status='save_error', pdb_id=pdb_id, message="Failed to save ligand SDF from complex.")

        logging.info(f"Successfully processed and saved all components for {pdb_id}.")
        return WorkerResult(status='success', pdb_id=pdb_id, message=None)

    except Exception as e:
        msg = f"An error occurred during file saving for {pdb_id}: {e}"
        logging.error(msg, exc_info=True)
        return WorkerResult(status='save_error', pdb_id=pdb_id, message=str(e))


def run_all_tasks_in_parallel() -> None:
    """Orchestrates the entire splitting process in parallel."""
    if not CONFIG or not PROJECT_ROOT:
        logging.critical("Configuration not loaded.")
        return

    # The list of PDB IDs is taken from the directory containing the docking results
    base_processing_dir = PROJECT_ROOT / CONFIG['split_task']['base_processing_dir']
    if not base_processing_dir.is_dir():
        logging.critical(f"Base processing directory does not exist: {base_processing_dir}")
        return

    logging.info(f"Scanning for PDB ID folders in: {base_processing_dir}")
    pdb_ids = [d.name for d in base_processing_dir.iterdir() if d.is_dir()]
    if not pdb_ids:
        logging.warning("No PDB ID subdirectories found in the base processing directory.")
        return

    logging.info(f"Found {len(pdb_ids)} PDB ID folders to process.")

    mp_config = CONFIG.get('multiprocessing', {})
    num_processes = mp_config.get('num_workers') or os.cpu_count() or 1
    chunk_size = mp_config.get('chunk_size', 1)

    logging.info(f"Starting parallel processing with {num_processes} workers...")
    with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker,
                              initargs=(log_queue, CONFIG, PROJECT_ROOT)) as pool:
        results: List[WorkerResult] = pool.map(process_pdb_id, pdb_ids, chunksize=chunk_size)

    # --- Detailed Summary ---
    logging.info("--- Splitting Process Complete ---")
    status_counts = {}
    failures = []
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in ('success', 'no_protein', 'no_complex'):
            failures.append(r)

    logging.info(f"Total directories processed: {len(results)}")
    for status, count in status_counts.items():
        log_level = logging.ERROR if 'error' in status else logging.INFO
        logging.log(log_level, f"  {status.capitalize()}: {count}")

    if failures:
        logging.error("--- List of Failures ---")
        for f in failures:
            logging.error(f"  PDB ID: {f['pdb_id']}, Status: {f['status']}, Message: {f['message']}")


# --- Main Entry Point ---
if __name__ == '__main__':
    listener = None
    log_queue = multiprocessing.Queue(-1)
    try:
        # Use a fixed path relative to the script for the config file
        config_file_path = Path(__file__).resolve().parent.parent / 'config.yaml'

        if not config_file_path.is_file():
            raise FileNotFoundError(f"Config file not found at expected path: {config_file_path}")

        load_config(config_file_path)

        setup_main_logging_handlers('split_log')
        listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
        listener.start()

        logging.info("--- Executing split.py in parallel mode ---")
        run_all_tasks_in_parallel()

    except FileNotFoundError as e:
        logging.critical(f"Essential file not found: {e}", exc_info=True)
    except Exception as e:
        logging.critical("An unexpected critical error occurred in the main process.", exc_info=True)
    finally:
        if listener:
            listener.stop()
        if log_queue:
            log_queue.close()
            log_queue.join_thread()