import random
from itertools import product
from typing import List, Dict

import numpy as np
import bittensor as bt

from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
from folding.utils.logger import logger


class HyperParameters:
    def __init__(self, exclude: Dict[str, str] = None) -> None:
        """Sample hyperparameters for protein folding for a specific pdb_id

        Args:
            exclude (List[str], optional): List of hyperparameters to exclude. Defaults to None.
                needs to be either ['FF','WATER'] to exclude a specific hyperparameter, or
                you can exclude a specific value(s) by passing {'FF': 'charmm27', 'BOX_TYPE': 'dodecahedron'}.
        """

        self.sampled_combinations: List[Dict[str, str]] = []
        self.all_combinations: List[Dict[str, str]] = []

        self.exclude: Dict[str, str] = exclude or {}

        if not (
            isinstance(self.exclude, dict)
            and all(isinstance(item, str) for item in self.exclude.values())
        ):
            raise ValueError(
                f"Exclude must be a Dict[str, str], received {self.exclude}"
            )

        # There are combinations of water and field that need to be upheld. If a field is removed, remove the entire class from search.
        fields = [field() for field in FORCEFIELD_REGISTRY.values()]

        initial_search: List[Dict[str, str]] = []

        for field in fields:
            FF: List[str] = field.forcefields
            WATER: List[str] = field.waters
            BOX = ["cube", "dodecahedron", "octahedron"]

            # We need to check if the FF they are asking for is actually a possible FF.
            # Also, we can only possibly check a water if we are excluding a FF.
            if "FF" in self.exclude:
                if self.exclude["FF"] not in FF:
                    continue

                if "WATER" in self.exclude and self.exclude["WATER"] not in WATER:
                    raise ValueError(
                        f"Selected water {self.exclude['WATER']} not found in possible waters for chosen {FF}: {WATER}"
                    )

            parameter_set = {
                "FF": FF,
                "WATER": WATER,
                "BOX": BOX,
            }

            filtered_parameter_set = self.create_parameter_space(
                parameter_set=parameter_set
            )
            self.all_combinations.extend(
                self.setup_combinations(parameter_set=filtered_parameter_set)
            )

            initial_search.append(field.recommended_configuration)

        # Shuffle the initial search and all combinations to ensure randomness.
        random.shuffle(initial_search)
        random.shuffle(self.all_combinations)

        # If we want to exclude a parameter from the search, we will remove the recommended configs.
        # This is simply because we want to avoid the edge cases and additional logic needed to handle this.
        if not len(self.exclude):
            np.random.shuffle(initial_search)  # inplace: to unbias the initial samples.
            self.all_combinations = initial_search + self.all_combinations

        self.TOTAL_COMBINATIONS = len(self.all_combinations)

    def create_parameter_space(
        self, parameter_set: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Make the parameter space based on the hyperparameters, and those to exclude.
        If exclude is an empty list, we do nothing.
        """

        for param in self.exclude.keys():
            try:
                parameter_set.pop(param)
            except KeyError:
                logger.error(
                    f"Parameter {param} not found in parameter set. Only FF, BOX_TYPE, and/or WATER are allowed."
                )

        return parameter_set

    def setup_combinations(
        self, parameter_set: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """
        Samples all possible combinations of hyperparameters for protein folding.
        Remove excluded hyperparameters from the list of possible combinations.
        """

        # Create a list of tuples, and then corresponds them in dictionary format.
        all_combinations = list(product(*parameter_set.values()))
        all_combinations = [
            {key: value for key, value in zip(parameter_set.keys(), combination)}
            for combination in all_combinations
        ]

        return all_combinations

    def sample_hyperparameters(self) -> Dict[str, str]:
        """Return a dictionary of pdb_id and sampled hyperparameters."""

        if len(self.sampled_combinations) == self.TOTAL_COMBINATIONS:
            return Exception(
                "All combinations have been sampled, reset the class to sample again."
            )

        # Sample hyperparameter set and add to set of sampled.
        sampled_combination = self.all_combinations.pop(0)
        self.sampled_combinations.append(sampled_combination)

        return sampled_combination
