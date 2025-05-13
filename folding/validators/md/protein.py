import os
import time
import glob
import base64
import shutil
import asyncio
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from openmm import app, unit
from pdbfixer import PDBFixer

from folding.base.simulation import OpenMMSimulation
from folding.store import Job
from folding.utils.logger import logger
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.ops import (
    OpenMMException,
    check_and_download_pdbs,
    check_if_directory_exists,
    write_pkl,
    load_pdb_file,
    save_files,
    create_velm,
    write_pdb_file,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Protein(OpenMMSimulation):
    PDB_RECORDS = ("ATOM", "ANISOU", "REMARK", "HETATM", "CONECT")

    @property
    def name(self):
        return self.protein_pdb.split(".")[0]

    def __init__(
        self,
        pdb_id: str,
        ff: str,
        water: str,
        box: Literal["cube", "dodecahedron", "octahedron"],
        config: Dict,
        system_kwargs: Dict,
        load_md_inputs: bool = False,
        epsilon: float = 1,  # percentage
        pdb_file_path: str | None = None,
        **kwargs,
    ) -> None:
        """The Protein class is responsible for handling the protein simulation.
        It attempts to setup the simulation environment to ensure that it can be run
        on the miner side. It also contains methods to validate the simulation outputs.

        Args:
            pdb_id (str): pdb_id of the protein, typically a 4 letter string.
            ff (str): the forcefield that will be used for simulation.
            water (str): water model that will be used for simulation.
            box (Literal): the shape of the box that will be used for simulation.
            config (Dict): bittensor config object.
            system_kwargs (Dict): system kwargs for the SimualtionConfig object.
            load_md_inputs (bool, optional): If we should load pre-comuted files for the protein. Defaults to False.
            epsilon (float, optional): The percentage improvement that must be achieved for this protein to be considered "better". Defaults to 1.
        """

        self.base_directory = os.path.join(str(ROOT_DIR), "data")
        self.pdb_id: str = pdb_id.lower()
        self.simulation_cpt = "em.cpt"
        self.simulation_pkl = f"config_{self.pdb_id}.pkl"
        self.pdb_file_path = pdb_file_path

        self.setup_filepaths()

        self.ff: str = ff
        self.box: Literal["cube", "dodecahedron", "octahedron"] = box
        self.water: str = water

        # The dict is saved as a string in the event, so it needs to be evaluated.
        if isinstance(system_kwargs, str):
            system_kwargs = eval(system_kwargs)

        self.system_config = SimulationConfig(
            ff=self.ff,
            water=self.water,
            box=self.box,
            seed=1337,
            **system_kwargs,
        )

        self.config = config
        self.simulation: app.Simulation = None
        self.input_files = [self.em_cpt_location]

        self.md_inputs = (
            self.read_and_return_files(filenames=self.input_files)
            if load_md_inputs
            else {}
        )

        # set to an arbitrarily high number to ensure that the first miner is always accepted.
        self.init_energy = 0
        self.pdb_complexity = defaultdict(int)
        self.epsilon = epsilon
        self.VALIDATOR_ID = os.getenv("VALIDATOR_ID")

    def setup_filepaths(self):
        # If a pdb file path is provided copy the file to the base directory.

        self.pdb_file = f"{self.pdb_id}.pdb"
        self.pdb_directory = os.path.join(self.base_directory, self.pdb_id)
        self.pdb_location = os.path.join(self.pdb_directory, self.pdb_file)
        if self.pdb_file_path:
            shutil.copy(
                self.pdb_file_path, os.path.join(self.pdb_directory, self.pdb_file)
            )

        self.validator_directory = os.path.join(self.pdb_directory, "validator")
        self.em_cpt_location = os.path.join(
            self.validator_directory, self.simulation_cpt
        )
        self.simulation_pkl_location = os.path.join(
            self.validator_directory, self.simulation_pkl
        )

        self.velm_array_pkl = os.path.join(
            self.pdb_directory, "velm_array_indicies.pkl"
        )

    @staticmethod
    async def from_job(job: Job, config: Dict):
        # Load_md_inputs is set to True to ensure that miners get files every query.
        protein = Protein(
            pdb_id=job.pdb_id,
            ff=job.system_config.ff,
            box=job.system_config.box,
            water=job.system_config.water,
            config=config,
            load_md_inputs=True,
            epsilon=job.epsilon,
            system_kwargs=job.system_config.system_kwargs.model_dump(),
        )

        try:
            protein.pdb_complexity = Protein._get_pdb_complexity(protein.pdb_location)
            protein._calculate_epsilon()

            protein.pdb_contents = protein.load_pdb_as_string(protein.pdb_location)

        except Exception as E:
            logger.error(f"from_job failed for {protein.pdb_id} with Exception {E}.")
            return None
        return protein

    @staticmethod
    def load_pdb_as_string(pdb_path: str) -> str:
        with open(pdb_path, "r") as f:
            return f.read()

    @staticmethod
    def _get_pdb_complexity(pdb_path):
        """Get the complexity of the pdb file by counting the number of atoms, residues, etc."""
        pdb_complexity = defaultdict(int)
        with open(pdb_path, "r") as f:
            for line in f.readlines():
                # Check if the line starts with any of the PDB_RECORDS
                for key in Protein.PDB_RECORDS:
                    if line.strip().startswith(key):
                        pdb_complexity[key] += 1
        return pdb_complexity

    async def setup_pdb_directory(self):
        # if directory doesn't exist, download the pdb file and save it to the directory
        if not os.path.exists(self.pdb_directory):
            os.makedirs(self.pdb_directory)

        if not os.path.exists(os.path.join(self.pdb_directory, self.pdb_file)):
            if not await check_and_download_pdbs(
                pdb_directory=self.pdb_directory,
                pdb_id=self.pdb_file,
                input_source=self.config.input_source,
                download=True,
                force=self.config.force_use_pdb,
            ):
                raise Exception(
                    f"Failed to download {self.pdb_file} to {self.pdb_directory}"
                )
            await self.fix_pdb_file()
        else:
            logger.info(
                f"PDB file {self.pdb_file} already exists in path {self.pdb_directory!r}."
            )

    def read_and_return_files(self, filenames: List) -> Dict:
        """Read the files and return them as a dictionary."""
        files_to_return = {}
        for filename in filenames:
            for file in glob.glob(os.path.join(self.validator_directory, filename)):
                try:
                    # A bit of a hack to load in the data correctly depending on the file ext
                    name = os.path.basename(file)
                    with open(file, "rb") as f:
                        if "cpt" in name:
                            files_to_return[name] = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        else:
                            files_to_return[name] = f.read()
                except Exception:
                    continue
        return files_to_return

    async def setup_simulation(self):
        """forward method defines the following:
        1. gather the pdb_id and setup the namings.
        2. setup the pdb directory and download the pdb file if it doesn't exist.
        3. check for missing files and generate the input files if they are missing.
        4. edit the necessary config files and add them to the synapse object self.md_inputs[file] = content
        4. save the files to the validator directory for record keeping.
        """

        logger.info(
            f"Launching {self.pdb_id} Protein Job with the following configuration\nff : {self.ff}\nbox : {self.box}\nwater : {self.water}"
        )

        await self.setup_pdb_directory()
        await self.generate_input_files()

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=self.validator_directory)

        # Read the files that should exist now based on generate_input_files.
        self.md_inputs = self.read_and_return_files(filenames=self.input_files)

        save_files(
            files=self.md_inputs,
            output_directory=self.validator_directory,
            write_mode="w",
        )

        self.pdb_complexity = Protein._get_pdb_complexity(self.pdb_location)
        self.init_energy = self.calc_init_energy()

        # Checking if init energy is nan
        if np.isnan(self.init_energy):
            raise OpenMMException(
                f"Failed to calculate initial energy for {self.pdb_id}"
            )

        self._calculate_epsilon()

    def __str__(self):
        return f"Protein(pdb_id={self.pdb_id}, ff={self.ff}, box={self.box}, water={self.water})"

    def __repr__(self):
        return self.__str__()

    async def fix_pdb_file(self):
        """
        Protein Data Bank (PDB or PDBx/mmCIF) files often have a number of problems
        that must be fixed before they can be used in a molecular dynamics simulation.

        The fixer will remove metadata that is contained in the header of the original pdb, and we might
        want to keep this. Therefore, we will rename the original pdb file to *_original.pdb and make a new
        pdb file using the PDBFile.writeFile method.

        Reference docs for the PDBFixer class can be found here:
            https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
        """

        fixer = PDBFixer(filename=self.pdb_location)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        write_pdb_file(
            pdb_location_path=self.pdb_location,
            topology=fixer.topology,
            positions=fixer.positions,
            suffix="_original",
        )

    # Function to generate the OpenMM simulation state.
    @OpenMMSimulation.timeit
    async def generate_input_files(self):
        """Generate_input_files method defines the following:
        1. Load the pdb file and create the simulation object.
        2. Minimize the energy of the system.
        3. Save the checkpoint file.
        4. Save the system config to the validator directory.
        5. Move all files except the pdb file to the validator directory.
        """
        logger.info(f"Loading PDB file from {self.pdb_location}")

        # Create simulation using absolute paths
        self.simulation, self.system_config = await asyncio.to_thread(
            self.create_simulation,
            load_pdb_file(pdb_file=self.pdb_location),
            self.system_config.get_config(),
            None,  # seed
            False,  # verbose
            True,  # initialize_with_solvent
        )

        # Get the altered pdb and write.
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        write_pdb_file(
            pdb_location_path=self.pdb_location,
            topology=self.simulation.topology,
            positions=positions,
            suffix="_before_solvent",
        )

        # load in information from the velm memory
        velm = create_velm(simulation=self.simulation)

        logger.info(f"Minimizing energy for pdb: {self.pdb_id} ...")
        start_time = time.time()
        await asyncio.to_thread(self.simulation.minimizeEnergy, maxIterations=100)
        logger.warning(f"Minimization took {time.time() - start_time:.4f} seconds")
        await asyncio.to_thread(self.simulation.step, 1000)

        # Save checkpoint using absolute path
        checkpoint_path = os.path.join(self.validator_directory, "em.cpt")
        check_if_directory_exists(output_directory=self.validator_directory)
        self.simulation.saveCheckpoint(checkpoint_path)

        # Save config using absolute path
        write_pkl(
            data=self.system_config, path=self.simulation_pkl_location, write_mode="wb"
        )

        # Move all non-PDB files to validator directory
        for file in os.listdir(self.pdb_directory):
            file_path = os.path.join(self.pdb_directory, file)
            if os.path.isfile(file_path) and not file.endswith(".pdb"):
                shutil.move(file_path, os.path.join(self.validator_directory, file))

        # Write velm array using absolute path
        write_pkl(
            data=velm,
            path=self.velm_array_pkl,
            write_mode="wb",
        )

    def get_miner_data_directory(self, hotkey: str):
        self.miner_data_directory = os.path.join(self.validator_directory, hotkey[:8])

    def check_gradient(self, check_energies: np.ndarray) -> True:
        """This method checks the gradient of the potential energy within the first
        WINDOW size of the check_energies array. Miners that return gradients that are too high,
        there is a *high* probability that they have not run the simulation as the validator specified.
        """
        WINDOW = 50  # Number of steps to calculate the gradient over
        GRADIENT_THRESHOLD = 10  # kJ/mol/nm

        mean_gradient = np.diff(check_energies[:WINDOW]).mean().item()
        return (
            mean_gradient <= GRADIENT_THRESHOLD
        )  # includes large negative gradients is passible

    def _calculate_epsilon(self):
        # TODO: Make this a better relationship?
        return self.epsilon

    def extract(self, filepath: str, names=["step", "default-name"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    def remove_pdb_directory(self):
        """Method to remove the pdb directory after the simulation is complete.
        Temp. method before we know what we want to keep.
        """
        shutil.rmtree(self.pdb_directory)

    def calc_init_energy(self) -> float:
        """Calculate the potential energy from an edr file using gmx energy.
        Args:
            output_dir (str): directory containing the edr file
            edr_name (str): name of the edr file
            xvg_name (str): name of the xvg file

        Returns:
            float: potential energy
        """

        return (
            self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            / unit.kilojoules_per_mole
        )
