import os
import time
import json
import random
import asyncio
import requests
import traceback
import concurrent.futures
from collections import defaultdict

from typing import Any

import psi4

from folding.miners.miner_mixin import MinerMixin
from folding.base.miner import BaseMinerNeuron
from folding.utils.logger import logger


class DFTMiner(BaseMinerNeuron, MinerMixin):
    def __init__(self, config=None):
        super().__init__(config=config)

        # TODO: There needs to be a timeout manager. Right now, if
        # the simulation times out, the only time the memory is freed is when the miner
        # is restarted, or sampled again.

        self.miner_data_path = os.path.join(self.project_path, "miner-data")
        self.base_data_path = os.path.join(
            self.miner_data_path, self.wallet.hotkey.ss58_address[:8]
        )
        self.local_db_address = os.getenv("RQLITE_HTTP_ADDR")
        self.simulations = DFTMiner.create_default_dict()

        self.max_workers = self.config.neuron.max_workers
        logger.info(f"🚀 Starting DFTMiner that handles {self.max_workers} workers 🚀")

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )  # remove one for safety

        self.mock = None
        self.generate_random_seed = lambda: random.randint(0, 1000)
        asyncio.run(self.start_rqlite())
        time.sleep(5)

    @staticmethod
    def create_default_dict():
        def nested_dict():
            return defaultdict(
                lambda: None
            )  # allows us to set the desired attribute to anything.

        return defaultdict(nested_dict)

    def response_to_dict(self, response) -> dict[str, Any]:
        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return {}

        columns = response["columns"]
        values = response["values"]
        data = [dict(zip(columns, row)) for row in values]
        return data

    def run(self):
        pass
