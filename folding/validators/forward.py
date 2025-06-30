from datetime import datetime
import os
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from async_timeout import timeout
from folding.validators.protein import Protein
from folding.utils.logging import log_event

import asyncio
from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
from folding.validators.hyperparameters import HyperParameters
from folding.utils.ops import (
    load_and_sample_random_pdb_ids,
    OpenMMException,
    RsyncException,
)
from folding.utils.logger import logger

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_config(config) -> Dict[str, str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """

    exclude_in_hp_search = {}

    if config.protein.ff is not None:
        exclude_in_hp_search["FF"] = config.protein.ff
    if config.protein.water is not None:
        exclude_in_hp_search["WATER"] = config.protein.water
    if config.protein.box is not None:
        exclude_in_hp_search["BOX"] = config.protein.box

    return exclude_in_hp_search


# TODO: We need to be able to create a bunch of different challenges.
async def create_new_challenge(self, exclude: List) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_md_challenge function.

    Args:
        exclude (List): list of pdb_ids to exclude from the search

    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """
    while True:
        forward_start_time = time.time()
        if self.RSYNC_EXCEPTION_COUNT > 10:
            self.config.protein.pdb_id = None
            self.config.protein.input_source = "rcsb"

        if self.config.protein.pdb_id is not None:
            pdb_id = self.config.protein.pdb_id
        else:
            pdb_id, input_source = load_and_sample_random_pdb_ids(
                root_dir=ROOT_DIR,
                filename="pdb_ids.pkl",
                input_source=self.config.protein.input_source,
                exclude=exclude,
            )
            self.config.protein.input_source = input_source

        # Perform a hyperparameter search until we find a valid configuration for the pdb
        logger.info(f"Attempting to prepare challenge for pdb {pdb_id}")
        event = await try_prepare_md_challenge(self, config=self.config, pdb_id=pdb_id)
        event["input_source"] = self.config.protein.input_source

        if event.get("validator_search_status"):
            return event
        else:
            # forward time if validator step fails
            event["hp_search_time"] = time.time() - forward_start_time

            # only log the event if the simulation was not successful
            log_event(self, event, failed=True)
            logger.debug(
                f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
            )
            exclude.append(pdb_id)


def create_random_modifications_to_system_config(config) -> Dict:
    """create modifications of the desired parameters.

    Looks at the base bittensor config and parses parameters that we have deemed to be
    valid for random sampling. This is to increase the problem space for miners.
    """
    sampler = lambda min_val, max_val: round(np.random.uniform(min_val, max_val), 2)

    system_kwargs = {"temperature": sampler(200, 400), "friction": sampler(0.9, 1.1)}

    for param in system_kwargs.keys():
        if config.protein[param] is not None:
            system_kwargs[param] = config.protein[param]
            continue

    return system_kwargs


async def try_prepare_md_challenge(self, config, pdb_id: str) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """
    logger.info(f"Searching parameter space for pdb {pdb_id}")

    exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

    # Create random modifications of parameters that are inside the function.
    system_kwargs = create_random_modifications_to_system_config(config=config)

    protein = None
    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        hp_sampler_time = time.time()

        event = {"hp_tries": tries}
        sampled_combination: Dict = hp_sampler.sample_hyperparameters()

        if config.protein.ff is not None:
            if (
                config.protein.ff is not None
                and config.protein.ff not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Forcefield {config.protein.ff} not found in FORCEFIELD_REGISTRY"
                )

        if config.protein.water is not None:
            if (
                config.protein.water is not None
                and config.protein.water not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Water {config.protein.water} not found in FORCEFIELD_REGISTRY"
                )

        hps = {
            "ff": config.protein.ff or sampled_combination["FF"],
            "water": config.protein.water or sampled_combination["WATER"],
            "box": config.protein.box or sampled_combination["BOX"],
        }

        protein = Protein(
            pdb_id=pdb_id,
            config=config.protein,
            system_kwargs=system_kwargs,
            **hps,
        )

        try:
            async with timeout(300):
                await protein.setup_simulation()

            if protein.init_energy > 0:
                raise ValueError(
                    f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                )

        except asyncio.TimeoutError as e:
            logger.info(f"Timeout occurred during simulation setup: {e}")
            event["validator_search_status"] = False
            tries = 10

        except OpenMMException as e:
            logger.info(f"OpenMMException occurred: init_energy is NaN {e}")
            event["validator_search_status"] = False

        except RsyncException as e:
            self.RSYNC_EXCEPTION_COUNT += 1
            event["validator_search_status"] = False

        except Exception as e:
            # full traceback
            logger.info(e)
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event["job_type"] = "SyntheticMD"
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            event["epsilon"] = protein.epsilon
            event["system_kwargs"] = system_kwargs
            event["s3_links"] = {
                "testing": "testing"
            }  # overwritten below if s3 logging is on.

            if "validator_search_status" not in event:
                if not config.s3.off:
                    try:
                        logger.info(f"Uploading to {self.handler.config.bucket_name}")
                        files_to_upload = {
                            "pdb": protein.pdb_location,
                            "cpt": os.path.join(
                                protein.validator_directory, protein.simulation_cpt
                            ),
                        }
                        location = os.path.join(
                            "inputs",
                            pdb_id,
                            self.validator_hotkey_reference,
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        )
                        s3_links = {}
                        for file_type, file_path in files_to_upload.items():
                            key = self.handler.put(
                                file_path=file_path,
                                location=location,
                                public=True,
                            )
                            s3_links[file_type] = os.path.join(
                                self.handler.output_url,
                                key,
                            )

                        event["s3_links"] = s3_links
                        event["validator_search_status"] = True  # simulation passed!
                        logger.success("✅✅ Simulation ran successfully! ✅✅")
                    except Exception as e:
                        logger.warning(f"Error uploading files to s3: {e}")
                        continue

                # break out of the loop if the simulation was successful
                break

            if tries == 10:
                logger.debug(f"Max tries reached for pdb_id {pdb_id} ❌❌")
                return event

    return event
