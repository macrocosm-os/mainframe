import os
import random
from typing import Any, Dict, Tuple

import bittensor as bt

import numpy as np
import pandas as pd
from openmm import app
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
from folding.base.evaluation import BaseEvaluator
from folding.base.simulation import OpenMMSimulation
from folding.utils import constants as c
from folding.utils.logger import logger
from folding.utils.ops import (
    ValidationError,
    create_velm,
    load_pdb_file,
    load_pkl,
    save_files,
    save_pdb,
    write_pkl,
    check_uniqueness,
)
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.protocol import IntermediateSubmissionSynapse
from folding.utils.s3_utils import DigitalOceanS3Handler


class SyntheticMDEvaluator(BaseEvaluator):
    def __init__(
        self,
        pdb_id: str,
        pdb_location: str,
        hotkey: str,
        state: str,
        seed: int,
        md_output: dict,
        basepath: str,
        system_config: SimulationConfig,
        velm_array_pkl_path: str,
        trajectory_s3_path: str,
        s3_handler: DigitalOceanS3Handler,
        **kwargs,
    ):
        self.pdb_id = pdb_id
        self.kwargs = kwargs
        self.md_simulator = OpenMMSimulation()
        self.pdb_location = pdb_location
        self.hotkey_alias = hotkey[:8]
        self.current_state = state
        self.miner_seed = seed
        self.md_output = md_output
        self.basepath = basepath

        self.system_config = system_config

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {
            k.split(".")[-1]: k for k, v in self.md_output.items() if len(v) > 0
        }
        self.miner_data_directory = os.path.join(self.basepath, self.hotkey_alias)
        self.velm_array_pkl_path = velm_array_pkl_path

        self.pdb_files = {}
        self.intermediate_checkpoint_files = {}
        self.miner_reported_energies = {}
        self.trajectory_s3_path = trajectory_s3_path
        self.s3_handler = s3_handler

    def process_md_output(self) -> bool:
        """Method to process molecular dynamics data from a miner and recreate the simulation.

        Args:
            md_output (dict): Data from the miner containing the output files.
            seed (int): Miner's seed used in the simulation.
            state (str): The state of the simulation.
            hotkey (str): The hotkey of the miner.
            basepath (str): The basepath of the validator to hold all the miner data.
            pdb_location (str): The location of the pdb file.

        Raises:
            ValidationError: Miner not running enough simulation steps

        Returns:
            bool: True if the simulation was successfully recreated, False otherwise.
        """

        required_files_extensions = ["cpt", "log"]

        if len(self.md_output.keys()) == 0:
            logger.warning(
                f"Miner {self.hotkey_alias} returned empty md_output... Skipping!"
            )
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                logger.error(f"Missing file with extension {ext} in md_output")
                return False

        # Save files so we can check the hash later.
        save_files(
            files=self.md_output,
            output_directory=self.miner_data_directory,
        )

        try:
            self.trajectory_path = os.path.join(
                self.miner_data_directory, "trajectory.dcd"
            )

            # download the trajectory from s3
            self.s3_handler.get(self.trajectory_s3_path, self.trajectory_path)

            # check if file exists
            if not os.path.exists(self.trajectory_path):
                logger.error(
                    f"Trajectory file {self.trajectory_path} does not exist... Skipping!"
                )
                return False

            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.
            logger.info(
                f"Recreating miner {self.hotkey_alias} simulation in state: {self.current_state}"
            )
            simulation, self.system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
                initialize_with_solvent=False,
            )

            checkpoint_path = os.path.join(
                self.miner_data_directory, f"{self.current_state}.cpt"
            )
            self.state_xml_path = os.path.join(
                self.miner_data_directory, f"{self.current_state}.xml"
            )
            self.log_file_path = os.path.join(
                self.miner_data_directory, self.md_outputs_exts["log"]
            )

            simulation.loadCheckpoint(checkpoint_path)

            self.log_file = pd.read_csv(self.log_file_path)
            self.log_step = self.log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(self.log_file) < c.MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (
                self.log_step - simulation.currentStep
            ) < c.MAX_SIMULATION_STEPS_FOR_EVALUATION:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(
                    self.miner_data_directory, f"{self.current_state}_old.cpt"
                )
                if os.path.exists(checkpoint_path):
                    logger.warning(
                        f"Miner {self.hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (
                        self.log_step - simulation.currentStep
                    ) < c.MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {self.hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            self.cpt_step = simulation.currentStep
            self.checkpoint_path = checkpoint_path

            self.steps_to_run = min(
                c.MAX_SIMULATION_STEPS_FOR_EVALUATION, self.log_step - self.cpt_step
            )

            # Create the state file here because it could have been loaded after MIN_SIMULATION_STEPS check
            simulation.saveState(self.state_xml_path)

            # Save the system config to the miner data directory
            self.system_config_path = os.path.join(
                self.miner_data_directory, f"miner_system_config_{self.miner_seed}.pkl"
            )

            self.final_miner_energies = self.log_file[
                (self.log_file['#"Step"'] > self.cpt_step)
                & (
                    self.log_file['#"Step"']
                    < self.cpt_step + c.MAX_SIMULATION_STEPS_FOR_EVALUATION
                )
            ]["Potential Energy (kJ/mole)"].values

            if not os.path.exists(self.system_config_path):
                write_pkl(
                    data=self.system_config,
                    path=self.system_config_path,
                    write_mode="wb",
                )

            miner_velm_data = create_velm(simulation=simulation)

            if not self.check_masses(miner_velm_data):
                raise ValidationError(
                    f"Miner {self.hotkey_alias} has modified the system in unintended ways... Skipping!"
                )
            self.number_of_checkpoints = (
                int(self.cpt_step // self.system_config.save_interval_checkpoint) - 1
            )
            if self.number_of_checkpoints < c.MAX_CHECKPOINTS_TO_VALIDATE:
                raise ValidationError(
                    f"Not enough intermediate checkpoints generated by miner. Need {c.MAX_CHECKPOINTS_TO_VALIDATE} but miner has generated {self.number_of_checkpoints}."
                )

        except ValidationError as E:
            logger.warning(f"{E}")
            return False

        except Exception as e:
            logger.error(f"Failed to recreate simulation: {e}")
            return False

        return True

    def get_reported_energy(self) -> float:
        """Get the energy from the simulation"""

        try:
            return float(np.median(self.final_miner_energies[-c.ENERGY_WINDOW_SIZE :]))
        except Exception as e:
            logger.error(f"Failed to get reported energy: {e}")
            return 0.0

    def check_masses(self, miner_velm_data) -> bool:
        """
        Check if the masses reported in the miner file are identical to the masses given
        in the initial pdb file. If not, they have modified the system in unintended ways.

        Reference:
        https://github.com/openmm/openmm/blob/53770948682c40bd460b39830d4e0f0fd3a4b868/platforms/common/src/kernels/langevinMiddle.cc#L11
        """

        validator_velm_data = load_pkl(self.velm_array_pkl_path, "rb")

        validator_masses = validator_velm_data["pdb_masses"]
        miner_masses = miner_velm_data["pdb_masses"]

        for i, (v_mass, m_mass) in enumerate(zip(validator_masses, miner_masses)):
            if v_mass != m_mass:
                logger.error(
                    f"Masses for atom {i} do not match. Validator: {v_mass}, Miner: {m_mass}"
                )
                return False
        return True

    def check_gradient(
        self, check_energies: np.ndarray, gradient_threshold: float
    ) -> Tuple[bool, float]:
        """This method checks the gradient of the potential energy within the first
        WINDOW size of the check_energies array. Miners that return gradients that are too high,
        there is a *high* probability that they have not run the simulation as the validator specified.
        """
        mean_gradient = np.diff(check_energies[: c.GRADIENT_WINDOW_SIZE]).mean().item()
        return (
            mean_gradient <= gradient_threshold,
            mean_gradient,
        )  # includes large negative gradients is passible

    def compare_state_to_cpt(
        self, state_energies: list, checkpoint_energies: list
    ) -> bool:
        """
        Check if the state file is the same as the checkpoint file by comparing the median of the first few energy values
        in the simulation created by the checkpoint and the state file respectively.
        """

        state_energies = np.array(state_energies)
        checkpoint_energies = np.array(checkpoint_energies)

        state_median = np.median(state_energies[: c.GRADIENT_WINDOW_SIZE])
        checkpoint_median = np.median(checkpoint_energies[: c.GRADIENT_WINDOW_SIZE])

        percent_diff = abs((state_median - checkpoint_median) / checkpoint_median) * 100

        if percent_diff > c.XML_CHECKPOINT_THRESHOLD:
            return False
        return True

    def select_stratified_checkpoints(
        self, num_checkpoints: int, num_samples: int
    ) -> list[int]:
        """Selects num_samples checkpoints from num_checkpoints at evenly spaced intervals."""

        # Create N evenly spaced bin edges, excluding the last edge (final checkpoint)
        edges = np.linspace(0, num_checkpoints, num_samples + 1, dtype=int)[:-1]

        # Sample one checkpoint randomly from each bin
        selected = [
            random.randint(start, max(start, end - 1))
            for start, end in zip(edges[:-1], edges[1:])
        ]
        return selected

    async def is_run_valid(
        self, validator=None, job_id=None, axon=None
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], str]:
        """
        Checks if the run is valid by evaluating a set of logical conditions:

        1. comparing the potential energy values between the current simulation and a reference log file.
        2. ensuring that the gradient of the minimization is within a certain threshold to prevent exploits.
        3. ensuring that the masses of the atoms in the simulation are the same as the masses in the original pdb file.
        4. If validator, job_id, and axon are provided, it also validates intermediate checkpoints from the miner.

        Args:
            validator: Optional validator instance to query for intermediate checkpoints
            job_id: Optional job ID to pass to get_intermediate_checkpoints
            axon: Optional axon to query for intermediate checkpoints

        Returns:
            Tuple[bool, dict, dict, str]: True if the run is valid, False otherwise.
                The two dicts contain the potential energy values from the current simulation and the reference log file.
                The string contains the reason for the run being invalid.
        """
        try:
            checked_energies_dict = {}
            miner_energies_dict = {}

            logger.info(f"Checking if run is valid for {self.hotkey_alias}...")
            logger.info(f"Checking final checkpoint...")
            (
                is_valid,
                checked_energies,
                miner_energies,
                result,
            ) = self.is_checkpoint_valid(
                checkpoint_path=self.checkpoint_path,
                current_cpt_step=self.cpt_step,
                steps_to_run=c.MAX_SIMULATION_STEPS_FOR_EVALUATION,
                gradient_threshold=c.GRADIENT_THRESHOLD_FINAL,
                checkpoint_num="final",
            )

            checked_energies_dict["final"] = checked_energies
            miner_energies_dict["final"] = miner_energies

            if not is_valid:
                return False, checked_energies_dict, miner_energies_dict, result

            # Check the intermediate checkpoints
            if validator is not None and job_id is not None and axon is not None:
                checkpoint_numbers = self.select_stratified_checkpoints(
                    num_checkpoints=self.number_of_checkpoints,
                    num_samples=c.MAX_CHECKPOINTS_TO_VALIDATE + 1,  # +1 for Final
                )

                # Get intermediate checkpoints from the miner
                intermediate_checkpoints: Dict[
                    str, Any
                ] = await self.get_intermediate_checkpoints(
                    validator=validator,
                    job_id=job_id,
                    axon=axon,
                    checkpoint_numbers=checkpoint_numbers,
                )

                if not intermediate_checkpoints:
                    return (
                        False,
                        checked_energies_dict,
                        miner_energies_dict,
                        "no-intermediate-checkpoints",
                    )

                # Validate each checkpoint
                for checkpoint_num, checkpoint_data in intermediate_checkpoints.items():
                    logger.info(f"Checking intermediate checkpoint {checkpoint_num}...")
                    if checkpoint_data is None:
                        return (
                            False,
                            checked_energies_dict,
                            miner_energies_dict,
                            f"checkpoint-is-none",
                        )

                    # Save the checkpoint data to a temporary file
                    temp_checkpoint_path = os.path.join(
                        self.miner_data_directory, f"intermediate_{checkpoint_num}.cpt"
                    )

                    self.intermediate_checkpoint_files[
                        checkpoint_num
                    ] = temp_checkpoint_path
                    with open(temp_checkpoint_path, "wb") as f:
                        f.write(checkpoint_data)

                    cpt_step = (
                        int(checkpoint_num) + 1
                    ) * self.system_config.save_interval_checkpoint

                    (
                        is_valid,
                        checked_energies,
                        miner_energies,
                        result,
                    ) = self.is_checkpoint_valid(
                        checkpoint_path=temp_checkpoint_path,
                        current_cpt_step=cpt_step,
                        steps_to_run=c.INTERMEDIATE_CHECKPOINT_STEPS,
                        checkpoint_num=checkpoint_num,
                        gradient_threshold=c.GRADIENT_THRESHOLD_INTERMEDIATE,
                    )

                    checked_energies_dict[checkpoint_num] = checked_energies
                    miner_energies_dict[checkpoint_num] = miner_energies

                    if not is_valid:
                        return False, checked_energies_dict, miner_energies_dict, result

                # Check if the miner's checkpoint is similar to the validator's checkpoint.
                miner_reported_energies = []
                checkpoint_length = len(
                    self.miner_reported_energies[str(checkpoint_numbers[0])]
                )
                for _, energy in self.miner_reported_energies.items():
                    miner_reported_energies.append(
                        energy[:checkpoint_length]
                    )  # final cpt is larger in length.

                if not check_uniqueness(
                    vectors=miner_reported_energies,
                    tol=c.MINER_CHECKPOINT_SIMILARITY_TOLERANCE,
                ):
                    logger.warning("Miner checkpoints not unique")
                    return (
                        False,
                        checked_energies_dict,
                        miner_energies_dict,
                        "miner-checkpoint-similarity",
                    )

                return True, checked_energies_dict, miner_energies_dict, "valid"

        except ValidationError as e:
            logger.warning(f"{e}")
            return False, {}, {}, e.message

        return True, checked_energies_dict, miner_energies_dict, "valid"

    def evaluate(self) -> bool:
        """Checks to see if the miner's data can be passed for validation"""
        if not self.process_md_output():
            return False

        # Check to see if we have a logging resolution of 10 or better, if not the run is not valid
        if (self.log_file['#"Step"'][1] - self.log_file['#"Step"'][0]) > 10:
            return False

        return True

    async def validate(
        self, validator=None, job_id=None, axon=None
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any], str]:
        """
        Validate the run by checking if it's valid and returning the appropriate metrics.

        Args:
            validator: Optional validator instance to query for intermediate checkpoints
            job_id: Optional job ID to pass to get_intermediate_checkpoints
            axon: Optional axon to query for intermediate checkpoints

        Returns:
            Tuple containing the median energy, checked energies, miner energies, and result message
        """
        (
            is_valid,
            checked_energies_dict,
            miner_energies_dict,
            result,
        ) = await self.is_run_valid(validator=validator, job_id=job_id, axon=axon)

        if not is_valid:
            # Return empty dictionaries to maintain consistency
            return 0.0, checked_energies_dict, miner_energies_dict, result

        # Use the final checkpoint's energy for the score
        if "final" in checked_energies_dict and checked_energies_dict["final"]:
            logger.success(f"Hotkey {self.hotkey_alias} passed validation!")
            final_energies = checked_energies_dict["final"]
            # Take the median of the last ENERGY_WINDOW_SIZE values
            median_energy = np.median(final_energies[-c.ENERGY_WINDOW_SIZE :])
            return median_energy, checked_energies_dict, miner_energies_dict, result
        else:
            # This should not happen if is_valid is True, but handle it just in case
            return (
                0.0,
                checked_energies_dict,
                miner_energies_dict,
                "missing-final-energies",
            )

    def _evaluate(self, data: Dict[str, Any]):
        pass

    def _validate(self):
        pass

    def get_ns_computed(self):
        """Calculate the number of nanoseconds computed by the miner."""

        return (self.cpt_step * self.system_config.time_step_size) / 1e3

    async def get_intermediate_checkpoints(
        self,
        validator: "Validator",
        job_id: str,
        axon: bt.Axon,
        checkpoint_numbers: list[int],
    ):
        """Get the intermediate checkpoints from the miner.

        Args:
            validator: The validator instance
            job_id: The job ID
            axon: The axon to query
            checkpoint_numbers: List of checkpoint numbers to retrieve
        """
        synapse = IntermediateSubmissionSynapse(
            job_id=job_id,
            checkpoint_numbers=checkpoint_numbers,
            pdb_id=self.pdb_id,
        )
        responses = await validator.dendrite.forward(
            synapse=synapse, axons=[axon], deserialize=True
        )
        return responses[0].cpt_files

    def name(self) -> str:
        return "SyntheticMD"

    def get_miner_log_file_energies(
        self, start_index: int, end_index: int
    ) -> np.ndarray:
        """Get the energies from the miner log file for a given range of steps."""
        miner_energies: np.ndarray = self.log_file[
            (self.log_file['#"Step"'] > start_index)
            & (self.log_file['#"Step"'] <= end_index)
        ]["Potential Energy (kJ/mole)"].values

        return miner_energies

    def calculate_rmsd(
        self, miner_trajectory, validator_trajectory, start_frame, end_frame
    ):
        """Calculate the RMSD between the miner and validator trajectories for every frame.

        Args:
            miner_trajectory: MDAnalysis Universe for miner trajectory
            validator_trajectory: MDAnalysis Universe for validator trajectory

        Returns:
            list[float]: List of RMSD values for each frame
        """
        # Get backbone atoms for both trajectories
        miner_bb = miner_trajectory.select_atoms("backbone")
        validator_bb = validator_trajectory.select_atoms("backbone")

        rmsds = []

        # Iterate through frames in both trajectories
        for miner_ts, validator_ts in zip(
            miner_trajectory.trajectory[start_frame:end_frame],
            validator_trajectory.trajectory,
        ):
            # Get coordinates for current frame
            miner_positions = miner_bb.positions.copy()
            validator_positions = validator_bb.positions.copy()

            # Calculate RMSD for current frame
            rmsd_value = rmsd(
                miner_positions, validator_positions, center=True, superposition=True
            )
            rmsds.append(rmsd_value)

        return rmsds

    def is_checkpoint_valid(
        self,
        checkpoint_path: str,
        current_cpt_step: int,
        steps_to_run: int = c.MIN_SIMULATION_STEPS,
        gradient_threshold: float = c.GRADIENT_THRESHOLD_FINAL,
        checkpoint_num: str = "final",
    ):
        """Validate a checkpoint by comparing energy values from simulation to reported values.

        This method avoids recreating simulation objects where possible by reusing them
        and resetting their state as needed.

        Args:
            checkpoint_path: Path to the checkpoint file to validate
            steps_to_run: Number of steps to run in the validation simulation

        Returns:
            Tuple of (is_valid, checked_energies, miner_energies, result_message)
        """
        # Create simulation once
        logger.info(
            f"Recreating simulation for {self.pdb_id}, checkpoint_num: {checkpoint_num} for state-based analysis..."
        )
        state_xml_path = os.path.join(
            self.miner_data_directory, f"{checkpoint_num}.xml"
        )

        # Create initial simulation
        simulation, _ = self.md_simulator.create_simulation(
            pdb=load_pdb_file(pdb_file=self.pdb_location),
            system_config=self.system_config.get_config(),
            seed=self.miner_seed,
            initialize_with_solvent=False,
        )

        # Load checkpoint
        simulation.loadCheckpoint(checkpoint_path)

        if current_cpt_step + steps_to_run > self.log_step:
            raise ValidationError(message="simulation-step-out-of-range")

        # Save state to XML file
        simulation.saveState(state_xml_path)

        simulation, _ = self.md_simulator.create_simulation(
            pdb=load_pdb_file(pdb_file=self.pdb_location),
            system_config=self.system_config.get_config(),
            seed=self.miner_seed,
            initialize_with_solvent=False,
        )
        simulation.loadState(state_xml_path)

        # Run state simulation and collect energies
        state_energies = []
        for _ in range(steps_to_run // 10):
            simulation.step(10)
            energy = (
                simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            )
            state_energies.append(energy)

        try:
            is_gradient_valid, mean_gradient = self.check_gradient(
                check_energies=np.array(state_energies),
                gradient_threshold=gradient_threshold,
            )
            if not is_gradient_valid:
                logger.warning(
                    f"❌ hotkey {self.hotkey_alias} failed state-gradient check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, mean_gradient: {mean_gradient} ... Skipping! ❌"
                )
                raise ValidationError(message="state-gradient")

            # Create a reporter for the checkpoint simulation
            current_state_logfile = os.path.join(
                self.miner_data_directory, f"check_{checkpoint_num}.log"
            )

            current_state_trajectory = os.path.join(
                self.miner_data_directory, f"check_{checkpoint_num}.dcd"
            )

            simulation, _ = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
                initialize_with_solvent=False,
            )

            # Load checkpoint
            simulation.loadCheckpoint(checkpoint_path)

            simulation.reporters.append(
                app.StateDataReporter(
                    current_state_logfile,
                    10,
                    step=True,
                    potentialEnergy=True,
                )
            )

            simulation.reporters.append(
                app.DCDReporter(
                    current_state_trajectory,
                    self.system_config.save_interval_trajectory,
                )
            )

            logger.info(
                f"Running {steps_to_run} steps. log_step: {self.log_step}, cpt_step: {current_cpt_step}"
            )

            # Run the checkpoint simulation
            simulation.step(steps_to_run)

            # Process results
            check_log_file = pd.read_csv(current_state_logfile)
            check_energies: np.ndarray = check_log_file[
                "Potential Energy (kJ/mole)"
            ].values

            max_step = current_cpt_step + steps_to_run

            miner_energies: np.ndarray = self.get_miner_log_file_energies(
                start_index=current_cpt_step, end_index=max_step
            )

            self.miner_reported_energies[checkpoint_num] = miner_energies

            if len(np.unique(check_energies)) == 1:
                logger.warning(
                    "All energy values in reproduced simulation are the same. Skipping!"
                )
                raise ValidationError(message="reprod-energies-identical")

            is_gradient_valid, mean_gradient = self.check_gradient(
                check_energies=np.array(check_energies),
                gradient_threshold=gradient_threshold,
            )
            if not is_gradient_valid:
                logger.warning(
                    f"❌ hotkey {self.hotkey_alias} failed cpt-gradient check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, mean_gradient: {mean_gradient} ... Skipping! ❌"
                )
                raise ValidationError(message="cpt-gradient")

            if not self.compare_state_to_cpt(
                state_energies=state_energies, checkpoint_energies=check_energies
            ):
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-checkpoint comparison for {self.pdb_id}, checkpoint_num: {checkpoint_num}, ... Skipping!"
                )
                raise ValidationError(message="state-checkpoint")

            # calculating absolute percent difference per step
            percent_diff = abs(
                ((check_energies - miner_energies) / miner_energies) * 100
            )
            median_percent_diff = np.median(percent_diff)

            if median_percent_diff > c.ANOMALY_THRESHOLD:
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed anomaly check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, with median percent difference: {median_percent_diff} ... Skipping!"
                )
                raise ValidationError(message="anomaly")

            # convert steps to frames
            start_frame = (
                current_cpt_step // self.system_config.save_interval_trajectory
            )
            end_frame = max_step // self.system_config.save_interval_trajectory

            check_universe = mda.Universe(self.pdb_location, current_state_trajectory)
            miner_universe = mda.Universe(self.pdb_location, self.trajectory_path)

            # Calculate RMSD between the trajectories for each frame
            rmsds = self.calculate_rmsd(
                miner_universe, check_universe, start_frame, end_frame
            )

            # Get median RMSD value
            median_rmsd = np.median(rmsds)

            if np.isnan(median_rmsd) or median_rmsd > 1:
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed trajectory RMSD check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, with median RMSD: {median_rmsd} ... Skipping!"
                )
                raise ValidationError(message="trajectory-rmsd")

            # Save the intermediate or final pdb file if the run is valid
            positions = simulation.context.getState(getPositions=True).getPositions()
            topology = simulation.topology
            pdb_output_path = os.path.join(
                self.miner_data_directory, f"{self.pdb_id}_{checkpoint_num}.pdb"
            )
            save_pdb(
                positions=positions, topology=topology, output_path=pdb_output_path
            )

            self.pdb_files[f"{self.pdb_id}_{checkpoint_num}"] = pdb_output_path

            return True, check_energies.tolist(), miner_energies.tolist(), "valid"

        except ValidationError as e:
            logger.warning(f"{e}")
            return False, [], [], e.message


class OrganicMDEvaluator(SyntheticMDEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def name(self) -> str:
        return "OrganicMD"


class SyntheticMLEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticML"


class OrganicMLEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicML"


EVALUATION_REGISTRY = {
    "SyntheticMD": SyntheticMDEvaluator,
    "OrganicMD": OrganicMDEvaluator,
    "SyntheticML": SyntheticMLEvaluator,
    "OrganicML": OrganicMLEvaluator,
}
