import torch
import random
import bittensor as bt
from typing import List
from folding.utils.logger import logger


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    include_serving_in_check: bool = True,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
        include_serving_in_check (bool): To only include uids that are serving in the check.
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if include_serving_in_check and not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.S[uid] > vpermit_tao_limit:
        return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(available_uids) < k:
        logger.warning(
            f"Requested {k} uids but only {len(available_uids)} were returned. To disable this, consider reducing neuron.sample_size"
        )
        k = len(available_uids)
    uids = torch.tensor(random.sample(available_uids, k))
    return uids


def get_all_miner_uids(
    metagraph, vpermit_tao_limit, include_serving_in_check: bool = True
) -> List[int]:
    """Returns all available miner uids from the metagraph.
    Returns:
        uids (List): All available miner uids.
        include_serving_in_check (bool): To only include miners that are actually serving in the check.
    """
    candidate_uids = []
    for uid in range(metagraph.n.item()):
        uid_is_available = check_uid_availability(
            metagraph=metagraph,
            uid=uid,
            vpermit_tao_limit=vpermit_tao_limit,
            include_serving_in_check=include_serving_in_check,
        )
        if uid_is_available:
            candidate_uids.append(uid)

    return candidate_uids
