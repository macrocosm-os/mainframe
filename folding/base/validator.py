# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2024 Macrocosmos AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import json
import copy
import torch
import asyncio
import argparse
import threading
import numpy as np
import bittensor as bt
from pathlib import Path
from datetime import datetime
from folding.utils.logger import logger
from typing import List, Optional

from folding.mock import MockDendrite
from folding.base.neuron import BaseNeuron
from folding.utils.ops import print_on_retry
from folding.utils.config import add_validator_args
from folding.organic.validator import OrganicValidator
from folding.registries.miner_registry import MinerRegistry

import tenacity

ROOT_DIR = Path(__file__).resolve().parents[2]


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        logger.info("Building validation weights.")
        self.scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            self._serve_axon()

        else:
            logger.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        self._organic_scoring: Optional[OrganicValidator] = None
        if self.config.neuron.organic_enabled:
            self._organic_scoring = OrganicValidator(
                validator=self,
                trigger=self.config.neuron.organic_trigger,
                trigger_frequency=self.config.neuron.organic_trigger_frequency,
            )

        else:
            logger.warning(
                "Organic scoring is not enabled. To enable, remove '--neuron.axon_off' and '--neuron.organic_enabled'"
            )

        self.load_and_merge_configs()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    def _serve_axon(self):
        """Serve axon to enable external connections"""
        validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(f"Serving validator IP of UID {validator_uid} to chain...")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor).start()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),  # Retry up to 3 times
        wait=tenacity.wait_fixed(1),  # Wait 1 second between retries
        retry=tenacity.retry_if_result(
            lambda result: result is False
        ),  # Retry if the result is False
        after=print_on_retry,
    )
    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        logger.info("Attempting to set weights...")

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = (
            torch.nn.functional.normalize(self.scores, p=1, dim=0).to("cpu").numpy()
        )

        logger.debug("raw_weights", raw_weights)
        logger.debug("raw_weight_uids", self.metagraph.uids)
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        logger.debug("processed_weights", processed_weights)
        logger.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        logger.debug("uint_weights", uint_weights)
        logger.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )

        logger.debug(result)
        return result[0]

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        logger.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced
                self.miner_registry.reset(miner_uid=uid)

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    async def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            logger.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, rewards
        ).to(self.device)

        # Matrix for alpha values
        scattered_alpha = torch.zeros_like(self.scores).to(self.device)

        for ii, uid in enumerate(uids_tensor):
            alpha = (
                self.config.neuron.positive_alpha
                if rewards[ii] > 0
                else self.config.neuron.negative_alpha
            )
            scattered_alpha[uid] = alpha

        # Update scores using exponential moving average
        self.scores: torch.FloatTensor = (
            scattered_alpha * scattered_rewards + (1 - scattered_alpha) * self.scores
        )

    def save_state(self):
        """Saves the state of the validator to a file."""
        logger.info("Saving validator state.")

        last_time_checked = (
            self.last_time_checked
            if hasattr(self, "last_time_checked")
            else datetime.now()
        )

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                "last_time_checked": last_time_checked,
            },
            self.config.neuron.full_path + "/state.pt",
        )

        self.miner_registry.save_registry(
            output_path=os.path.join(self.config.neuron.full_path, "miner_registry.pkl")
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        try:
            state = torch.load(self.config.neuron.full_path + "/state.pt")
            self.step = state["step"]
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
            self.last_time_checked = (
                state["last_time_checked"]
                if "last_time_checked" in state
                else datetime.now()
            )
            logger.info("Loaded previously saved validator state information.")

        except FileNotFoundError:
            logger.info(
                "Previous validator state not found... Weight copying the average of the network."
            )

            self.scores = self.get_chain_weights()
            self.step = 1

        try:
            logger.info("Loading miner registry.")
            self.miner_registry = MinerRegistry.load_registry(
                input_path=os.path.join(
                    self.config.neuron.full_path, "miner_registry.pkl"
                )
            )

        except Exception as e:
            logger.error(f"Error loading miner registry: {e}")

    def get_chain_weights(self) -> torch.Tensor:
        """Obtain the stake weighted average of all validator weights on chain."""
        valid_indices = np.where(self.metagraph.validator_permit)[0]
        valid_weights = self.metagraph.weights[valid_indices]
        valid_stakes = self.metagraph.S[valid_indices]
        normalized_stakes = valid_stakes / np.sum(valid_stakes)

        weights = torch.tensor(np.dot(normalized_stakes, valid_weights)).to(self.device)

        # normalize weights to be between 0 and 1
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        return weights

    def load_config_json(self):
        config_json_path = os.path.join(
            str(ROOT_DIR), "folding/utils/config_input.json"
        )
        with open(config_json_path, "r") as file:
            config = json.load(file)
        return config

    def load_and_merge_configs(self):
        json_config = self.load_config_json()
        self.config.protein.update(json_config)
