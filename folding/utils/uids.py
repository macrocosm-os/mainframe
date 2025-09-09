import bittensor as bt
from typing import List


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
