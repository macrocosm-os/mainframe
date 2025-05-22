# Class that holds all the S3 handling, job searching and other utilities

import os
import json
import time
import requests
import traceback
import hashlib

from typing import Dict, List, Union

from folding.utils.logger import logger
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.reporters import (
    ExitFileReporter,
    LastTwoCheckpointsReporter,
    SequentialCheckpointReporter,
    ProteinStructureReporter,
)

import openmm.app as app


class MinerMixin:
    def __init__(self):
        pass

    def fetch_sql_job_details(
        self, columns: List[str], job_id: str, local_db_address: str
    ) -> Dict:
        """
        Fetches job records from a SQLite database with given column details and a specific job_id.

        Parameters:
            columns (list): List of column names to retrieve from the database.
            job_id (str): The identifier for the job to fetch.
            db_path (str): Path to the SQLite database file.

        Returns:
            dict: A dictionary mapping job_id to its details as specified by the columns list.
        """

        logger.info("Fetching job details from the sqlite database")

        full_local_db_address = f"http://{local_db_address}/db/query"
        columns_to_select = ", ".join(columns)
        query = f"""SELECT job_id, {columns_to_select} FROM jobs WHERE job_id = '{job_id}'"""

        try:
            response = requests.get(
                full_local_db_address,
                params={"q": query, "level": "strong"},
                timeout=10,
            )
            response.raise_for_status()

            data: dict = self.response_to_dict(response=response)
            logger.info(f"data response: {data}")

            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch job details {e}")
            return

    def get_simulation_hash(self, initial_system_hash: str, system_config: Dict) -> str:
        """Creates a simulation hash based on the initial_system_hash and the system_config given.

        Returns:
            str: first 6 characters of a sha256 hash
        """
        system_hash = initial_system_hash
        for key, value in system_config.items():
            system_hash += str(key) + str(value)

        hash_object = hashlib.sha256(system_hash.encode("utf-8"))
        return hash_object.hexdigest()[:6]

    def download_gjp_input_files(
        self,
        output_dir: str,
        identifier: str,
        s3_links: dict[str, str],
    ) -> bool:
        """Downloads input files required from the GJP (Global Job Pool) from S3 storage.

        Args:
            output_dir (str): Directory path where downloaded files will be saved
            identifier (str): Identifier for the job
            s3_links (dict[str, str]): Dictionary mapping file types to their S3 URLs

        Returns:
            bool: True if all files were downloaded successfully, False if any download failed

        The function:
        1. Creates the output directory if it doesn't exist
        2. Downloads each file in chunks to conserve memory
        3. Names downloaded files as {pdb_id}.{file_type}
        """

        def stream_download(url: str, output_path: str):
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        for key, url in s3_links.items():
            output_path = os.path.join(output_dir, f"{identifier}.{key}")
            try:
                stream_download(url=url, output_path=output_path)
            except Exception as e:
                logger.error(f"Failed to download file {key} with error: {e}")
                return False
        return True

    def check_if_job_was_worked_on(
        self, job_id: str, local_db_address: str
    ) -> tuple[bool, str, dict]:
        """Check if a job has been previously worked on or is currently being processed.

        Parameters:
            job_id (str): The unique identifier for the job to check.

        Returns:
            tuple[bool, str, dict]: A tuple containing:
                - Whether the job has been worked on (bool)
                - The condition of the job (str)
                - Event dictionary with job details (dict)
        """

        columns = ["pdb_id", "system_config"]

        # query your LOCAL rqlite db to get pdb_id
        sql_job_details = self.fetch_sql_job_details(
            columns=columns, job_id=job_id, local_db_address=local_db_address
        )[0]

        if len(sql_job_details) == 0:
            logger.warning(f"Job ID {job_id} not found in the database.")
            return False, "job_not_found", {}

        # If we are already running a process with the same identifier, return intermediate information
        logger.info(f"⌛ Checking for job: {job_id} ⌛")

        event = self.create_default_dict()
        event["identifier"] = job_id

        gjp_config = json.loads(sql_job_details["system_config"])
        event["gjp_config"] = gjp_config

        _hash = self.get_simulation_hash(
            initial_system_hash=job_id, system_config=gjp_config
        )
        event["hash"] = _hash

        output_dir = os.path.join(self.base_data_path, job_id, _hash)
        gjp_config_filepath = os.path.join(output_dir, f"config_{job_id}.pkl")

        event["output_dir"] = output_dir
        event["gjp_config_filepath"] = gjp_config_filepath

        event = self.check_and_remove_simulations(event=event)
        if _hash in self.simulations:
            return True, "running_simulation", event

        # check if any of the simulations have finished

        submitted_job_is_unique = self.is_unique_job(
            system_config_filepath=gjp_config_filepath
        )

        if not submitted_job_is_unique:
            return True, "found_existing_data", event

        return False, "job_not_worked_on", event

    def add_active_jobs_from_db(self, query: str, local_db_address: str) -> int:
        """
        Fetch active jobs from the database and add them to the simulation executor.

        Returns:
            int: Number of jobs added to the executor

        # We need columns that identify the job and contain essential configuration
        # The query would need to look something like this:
        columns_to_select = "geometry, system_config, priority, s3_links"
        query = "
            SELECT job_id, {columns_to_select} FROM jobs
                    WHERE active = 1
                    ORDER BY priority DESC, created_at DESC
                    "

        """
        if not local_db_address:
            logger.warning(
                "No local database address configured, cannot add active jobs"
            )
            return 0

        # Calculate how many slots are available
        available_slots = self.max_workers - len(self.simulations)
        if available_slots <= 0:
            logger.info("No available worker slots for new jobs")
            return 0

        # Query the database for active jobs that are not already being processed
        full_local_db_address = f"http://{local_db_address}/db/query"

        try:
            response = requests.get(
                full_local_db_address,
                params={"q": query, "level": "strong"},
                timeout=10,
            )
            response.raise_for_status()

            data = self.response_to_dict(response=response)
            if not data or len(data) == 0:
                logger.info("No active jobs found in database")
                return 0

            logger.info(f"Number of active jobs in gjp: {len(data)}")

            # Keep track of how many jobs we've added
            jobs_added = 0

            # Add each job to the simulation executor if not already being processed
            for job in data:
                has_worked_on_job, _, event = self.check_if_job_was_worked_on(
                    job_id=job.get("job_id"), local_db_address=local_db_address
                )
                if has_worked_on_job:
                    logger.info(
                        f"Job {job.get('job_id')} is already being worked on or has been worked on before"
                    )
                    continue

                job_id = job.get("job_id")
                system_config_json = job.get("system_config")
                s3_links = job.get("s3_links")

                if not job_id or not system_config_json:
                    logger.warning(f"Incomplete job data: {job}")
                    continue

                # Generate a unique hash for this job to check if it's already running
                try:
                    system_config = json.loads(system_config_json)
                    _hash = self.get_simulation_hash(
                        initial_system_hash=job_id, system_config=system_config
                    )

                    # Skip if this simulation is already running
                    if _hash in self.simulations:
                        logger.info(
                            f"Simulation for job {job_id} (hash: {_hash}) is already running"
                        )
                        continue

                    # Create an output directory for this job
                    output_dir = os.path.join(self.base_data_path, job_id, _hash)
                    os.makedirs(output_dir, exist_ok=True)

                    success = self.download_gjp_input_files(
                        identifier=job_id,
                        output_dir=output_dir,
                        s3_links=json.loads(s3_links),
                    )
                    if not success:
                        logger.error(
                            f"Failed to download GJP input files for job {job_id}"
                        )
                        continue

                    # Create simulation config
                    simulation_config = self.get_simulation_config(
                        gjp_config=system_config,
                        system_config_filepath=os.path.join(
                            output_dir, f"config_{job_id}.pkl"
                        ),
                    )

                    # Add the job to the simulation executor
                    event = {"condition": "loading_from_db"}
                    self.create_simulation_from_job(
                        output_dir=output_dir,
                        identifier=job_id,
                        key=_hash,
                        system_config=simulation_config,
                        event=event,
                    )

                    jobs_added += 1
                    logger.success(f"Added job {job_id} from database to executor")

                    # Stop if we've reached our limit
                    if jobs_added >= available_slots:
                        break

                except Exception:
                    logger.error(
                        f"Failed to add job {job_id} from database to executor: {traceback.format_exc()}"
                    )
                    continue

            logger.info(f"Added {jobs_added} jobs from database to simulation executor")
            return jobs_added

        except requests.RequestException as e:
            logger.error(f"Failed to fetch active jobs from database: {e}")
            return 0

    def create_simulation_from_job(
        self,
        output_dir: str,
        identifier: str,
        key: str,
        system_config: Union[SimulationConfig, dict],
        event: Dict,
    ):
        """Create a simulation from a job

        Args:
            output_dir (str): The output directory for the simulation
            identifier (str): The identifier for the job
            key (str): The key for the simulation
            system_config (Union[SimulationConfig, dict]): The system configuration for the simulation
            event (Dict): The event for the simulation
        """
        # Submit job to the executor
        simulation_manager = SimulationManager(
            identifier=identifier,
            output_dir=output_dir,
            system_config=system_config.model_dump()
            if isinstance(system_config, SimulationConfig)
            else system_config,
            seed=system_config.seed,
        )

        future = self.executor.submit(
            simulation_manager.run,
            self.config.mock or self.mock,  # self.mock is inside of MockFoldingMiner
        )

        self.simulations[key]["identifier"] = identifier
        self.simulations[key]["executor"] = simulation_manager
        self.simulations[key]["future"] = future
        self.simulations[key]["output_dir"] = simulation_manager.output_dir
        self.simulations[key]["queried_at"] = time.time()

        logger.success(f"✅ New job {identifier} submitted to job executor ✅ ")

        event["condition"] = "new_simulation"
        event["start_time"] = time.time()


class SimulationManager:
    def __init__(
        self,
        task_type: str,
        identifier: str,
        output_dir: str,
        seed: int,
        system_config: dict,
    ) -> None:
        self.task_type = task_type
        self.identifier = identifier
        self.state: str = None
        self.seed = seed

    def _setup(self, task_type: str):
        if task_type == "MD":
            self.pdb_obj = app.PDBFile(os.path.join(output_dir, f"{pdb_id}.pdb"))

            self.state_file_name = f"{pdb_id}_state.txt"
            self.seed_file_name = f"{pdb_id}_seed.txt"
            self.simulation_steps: dict = system_config["simulation_steps"]
            self.system_config = SimulationConfig(**system_config)

            self.output_dir = output_dir
            self.start_time = time.time()

            self.cpt_file_mapper = {
                "nvt": f"{output_dir}/{self.pdb_id}.cpt",
                "npt": f"{output_dir}/nvt.cpt",
                "md_0_1": f"{output_dir}/npt.cpt",
            }

            self.STATES = ["nvt", "npt", "md_0_1"]
            self.CHECKPOINT_INTERVAL = 10000
            self.STATE_DATA_REPORTER_INTERVAL = 10
            self.EXIT_REPORTER_INTERVAL = 10

    def create_empty_file(self, file_path: str):
        # For mocking
        with open(file_path, "w") as f:
            pass

    def write_state(self, state: str, state_file_name: str, output_dir: str):
        with open(os.path.join(output_dir, state_file_name), "w") as f:
            f.write(f"{state}\n")

    def run(
        self,
    ):
        """run method to handle the processing of generic simulations.

        Args:
            simulations (Dict): state_name : OpenMMSimulation object dictionary
            suppress_cmd_output (bool, optional): Defaults to True.
        """
        logger.info(f"Running simulation for protein: {self.pdb_id}")
        simulations = self.configure_commands(
            seed=self.seed, system_config=copy.deepcopy(self.system_config)
        )
        logger.info(f"Simulations: {simulations}")

        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir)

        # Write the seed so that we always know what was used.
        with open(self.seed_file_name, "w") as f:
            f.write(f"{self.seed}\n")

        try:
            for state, simulation in simulations.items():
                logger.info(f"Running {state} commands")

                self.write_state(
                    state=state,
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir,
                )

                simulation.loadCheckpoint(self.cpt_file_mapper[state])
                simulation.step(self.simulation_steps[state])
                simulation.saveCheckpoint(f"{self.output_dir}/{state}.cpt")
                # TODO: Add a Mock pipeline for the new OpenMM simulation here.

            logger.success(f"✅ Finished simulation for protein: {self.pdb_id} ✅")

            state = "finished"
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, None

        # This is the exception that is raised when the simulation fails.
        except mm.OpenMMException as e:
            state = "failed"
            error_info = {
                "type": "OpenMMException",
                "message": str(e),
                "traceback": traceback.format_exc(),  # This is the traceback of the exception
            }
            try:
                platform = mm.Platform.getPlatformByName("CUDA")
                error_info["cuda_version"] = platform.getPropertyDefaultValue(
                    "CudaCompiler"
                )
            except:
                error_info["cuda_version"] = "Unable to get CUDA information"
            finally:
                self.write_state(
                    state=state,
                    state_file_name=self.state_file_name,
                    output_dir=self.output_dir,
                )
                return state, error_info

        # Generic Exception
        except Exception as e:
            state = "failed"
            error_info = {
                "type": "UnexpectedException",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            self.write_state(
                state=state,
                state_file_name=self.state_file_name,
                output_dir=self.output_dir,
            )
            return state, error_info

    def get_state(self) -> str:
        """get_state reads a txt file that contains the current state of the simulation"""
        with open(os.path.join(self.output_dir, self.state_file_name), "r") as f:
            lines = f.readlines()
            return (
                lines[-1].strip() if lines else None
            )  # return the last line of the file

    def get_seed(self) -> str:
        with open(os.path.join(self.output_dir, self.seed_file_name), "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else None

    def configure_commands(
        self, seed: int, system_config: SimulationConfig
    ) -> Dict[str, List[str]]:
        state_commands = {}

        for state, simulation_steps in self.simulation_steps.items():
            simulation, _ = OpenMMSimulation().create_simulation(
                pdb=self.pdb_obj,
                system_config=system_config.get_config(),
                seed=seed,
            )
            simulation.reporters.append(
                LastTwoCheckpointsReporter(
                    file_prefix=f"{self.output_dir}/{state}",
                    reportInterval=self.CHECKPOINT_INTERVAL,
                )
            )

            simulation.reporters.append(
                ExitFileReporter(
                    filename=f"{self.output_dir}/{state}",
                    reportInterval=self.EXIT_REPORTER_INTERVAL,
                    file_prefix=state,
                )
            )

            # Calculate the starting checkpoint counter based on previous states
            starting_counter = 0
            if state in self.simulation_steps:
                state_index = list(self.simulation_steps.keys()).index(state)
                previous_states = list(self.simulation_steps.keys())[:state_index]

                # Sum the number of checkpoints for all previous states
                for prev_state in previous_states:
                    # Calculate how many checkpoints were created in the previous state
                    prev_checkpoints = int(
                        self.simulation_steps[prev_state] / self.CHECKPOINT_INTERVAL
                    )
                    starting_counter += prev_checkpoints

            simulation.reporters.append(
                SequentialCheckpointReporter(
                    file_prefix=f"{self.output_dir}/",
                    reportInterval=self.CHECKPOINT_INTERVAL,
                    checkpoint_counter=starting_counter,
                )
            )

            simulation.reporters.append(
                ProteinStructureReporter(
                    file=f"{self.output_dir}/{state}.log",
                    reportInterval=self.STATE_DATA_REPORTER_INTERVAL,
                    step=True,
                    potentialEnergy=True,
                    reference_pdb=os.path.join(self.output_dir, f"{self.pdb_id}.pdb"),
                    speed=True,
                )
            )
            state_commands[state] = simulation

        return state_commands
