from itertools import chain
from typing import List, Dict, Any
from dataclasses import field
from pydantic import BaseModel

import math
from statistics import mean

import folding.utils.constants as c
from folding.utils.logger import logger
from folding.utils.ops import write_pkl, load_pkl
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


class TaskMetrics(BaseModel):
    """Holds metrics for a specific task"""

    credibility: float = c.STARTING_CREDIBILITY
    credibilities: List[List[float]] = field(default_factory=list)
    credibility_over_time: List[float] = field(default_factory=list)


class MinerData(BaseModel):
    """Holds all data for a miner"""

    overall_credibility: float = c.STARTING_CREDIBILITY
    tasks: Dict[str, TaskMetrics] = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)


class MinerRegistry:
    """Class for handling miner scores, credibilities, and other attributes."""

    def __init__(self, miner_uids: List[int]):
        self.tasks: List[str] = list(EVALUATION_REGISTRY.keys())
        self.registry: Dict[int, MinerData] = {}

        for miner_uid in miner_uids:
            self._add_miner_to_registry(miner_uid)

    def _add_miner_to_registry(self, miner_uid: int) -> None:
        """Adds a miner to the registry with default values."""
        miner_data = MinerData()
        miner_data.tasks = {task: TaskMetrics() for task in self.tasks}
        self.registry[miner_uid] = miner_data
        self._initialize_miner_logs(miner_uid)

    def _get_or_create_miner(self, miner_uid: int) -> MinerData:
        """Gets existing miner or creates new entry if not exists."""
        if miner_uid not in self.registry:
            self._add_miner_to_registry(miner_uid)
        return self.registry[miner_uid]

    def _initialize_miner_logs(self, miner_uid: int) -> None:
        """Sets the logs for a miner used for the evaluation/validation pipelines"""

        initial_logs = {
            "can_process": False,
            "reported_energy": 0,
            "seed": None,
            "files": {},
            "process_md_output_time": 0,
            "is_valid": False,
            "checked_energies": None,
            "miner_energies": None,
            "rmsd": 0,
            "is_run_valid_time": 0,
            "ns_computed": 0,
            "reason": "",
            "is_duplicate": False,
            "axon": None,
        }

        self.registry[miner_uid].logs = initial_logs

    def add_credibilities(
        self, miner_uid: int, task: str, credibilities: List[float]
    ) -> None:
        """Adds credibilities to the miner registry."""
        if task not in self.tasks:
            raise ValueError(f"Invalid task: {task}")
        if not all(isinstance(c, (int, float)) for c in credibilities):
            raise ValueError("All credibilities must be numeric")

        miner = self._get_or_create_miner(miner_uid)
        miner.tasks[task].credibilities.append(credibilities)

    def get_credibilities(self, miner_uid: int, task: str = None) -> List[float]:
        """Returns the credibilities for a miner and task."""
        if task not in self.tasks:
            raise ValueError(f"Invalid task: {task}")

        miner = self._get_or_create_miner(miner_uid)

        if task is None:
            return miner.overall_credibility
        return miner.tasks[task].credibility

    def update_credibility(self, miner_uid: int, task: str) -> None:
        """Updates the credibility of a miner for a specific task."""
        if task not in self.tasks:
            raise ValueError(f"Invalid task: {task}")

        miner = self._get_or_create_miner(miner_uid)
        task_metrics = miner.tasks[task]

        # Flatten credibilities list and process each credibility
        task_credibilities = list(chain.from_iterable(task_metrics.credibilities))

        current_credibility = task_metrics.credibility
        for cred in task_credibilities:
            alpha = (
                c.CREDIBILITY_ALPHA_POSITIVE
                if cred > 0
                else c.CREDIBILITY_ALPHA_NEGATIVE
            )
            current_credibility = round(
                (alpha * cred + (1 - alpha) * current_credibility), 3
            )

        task_metrics.credibility = current_credibility
        task_metrics.credibility_over_time.append(current_credibility)
        task_metrics.credibilities.clear()  # Reset credibilities

        # Update overall credibility
        all_credibilities = [t.credibility for t in miner.tasks.values()]
        miner.overall_credibility = round(mean(all_credibilities), 3)

    def get_validation_probability(self, miner_uid: int, task: str) -> float:
        """Returns the probability of validating a miner's work."""
        miner_credibility = self.get_credibilities(miner_uid=miner_uid, task=task)
        validation_probability = (
            1
            if miner_credibility <= 0.5
            else math.exp(-4.605 * (miner_credibility - 0.5))
        )
        return validation_probability

    def reset(self, miner_uid: int) -> None:
        """Resets all metrics for a miner."""
        self._add_miner_to_registry(miner_uid)

    def get_all_miner_logs(self) -> Dict[int, Dict[str, Any]]:
        """Returns all logs for all miners and resets them."""
        all_miner_logs = {}

        for miner_uid, miner_data in self.registry.items():
            if not hasattr(miner_data, "logs") or miner_data.logs is None:
                self._initialize_miner_logs(miner_uid)
            all_miner_logs[miner_uid] = miner_data.logs

        return all_miner_logs

    def reset_miner_logs(self) -> None:
        """Resets the logs for all miners."""
        for miner_uid in self.registry:
            self._initialize_miner_logs(miner_uid)

    @classmethod
    def load_registry(cls, input_path: str) -> "MinerRegistry":
        try:
            return load_pkl(path=input_path, read_mode="rb")
        except Exception as e:
            raise e

    def save_registry(self, output_path: str) -> None:
        write_pkl(data=self, path=output_path, write_mode="wb")
