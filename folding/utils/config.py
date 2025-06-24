import os
import argparse
import bittensor as bt
from folding.utils.logger import setup_file_logging, add_events_level


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            "~/.bittensor/miners",
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        add_events_level()
        setup_file_logging(
            os.path.join(config.neuron.full_path, "events.log"),
            config.neuron.events_retention_size,
        )


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=25)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--neuron.metagraph_resync_length",
        type=int,
        help="The number of blocks until metagraph is resynced.",
        default=100,
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.mock",
        action="store_true",
        help="Dry run.",
        default=False,
    )

    parser.add_argument(
        "--protein.pdb_id",
        type=str,
        help="PDB ID for protein folding.",  # defaults to None
        default=None,
    )

    parser.add_argument(
        "--protein.ff",
        type=str,
        help="Force field for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.water",
        type=str,
        help="Water used for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.box",
        type=str,
        help="Box type for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.save_interval",
        type=int,
        help="How many steps before saving values to files.",
        default=2000,
    )

    parser.add_argument(
        "--protein.temperature",
        type=float,
        help="Temperature of the simulation. Typically between 200-400K",
        default=None,
    )

    parser.add_argument(
        "--protein.friction",
        type=float,
        help="Friction of the simulation. Typically between 0.9-1.1",
        default=None,
    )

    parser.add_argument(
        "--protein.max_steps",
        type=int,
        help="Maximum number of steps for protein folding.",
        default=750000,
    )

    parser.add_argument(
        "--protein.npt_steps",
        type=int,
        help="Number of steps run in npt stage of protein simulation. If None, it is calculated via max_steps.",
        default=None,
    )

    parser.add_argument(
        "--protein.nvt_steps",
        type=int,
        help="Number of steps run in nvt stage of protein simulation. If None, it is calculated via max_steps.",
        default=None,
    )

    parser.add_argument(
        "--protein.num_steps_to_save",
        type=int,
        help="NOT IN USE: Maximum number of steps to save during the energy minimization routine (set by validators for miners).",
        default=100,
    )

    parser.add_argument(
        "--protein.suppress_cmd_output",
        action="store_true",
        help="If set, we suppress the text output of terminal commands to reduce terminal clutter.",
        default=True,
    )

    parser.add_argument(
        "--protein.verbose",
        action="store_true",
        help="If set, any errors on terminal commands will be reported in logs.",
        default=True,
    )

    parser.add_argument(
        "--protein.force_use_pdb",
        action="store_true",
        help="If True, we will attempt to fold a protein that has missing atoms.",
        default=True,
    )

    parser.add_argument(
        "--protein.seed",
        type=int,
        help="Set a random seed to initialize the simulation for miners.",
        default=None,
    )

    parser.add_argument(
        "--protein.input_source",
        type=str,
        help="Specifies the input source for selecting a new protein for simulation.",
        default="rcsb",
        choices=["rcsb", "pdbe"],
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="25 MB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    parser.add_argument(
        "--mdrun_args.ntmpi",
        type=str,
        help="Controls the number of processes that are used to run the simulation",
        default=1,
    )

    parser.add_argument(
        "--s3.off",
        action="store_true",
        help="If set to True, then S3 logging is turned off.",
        default=False,
    )

    parser.add_argument(
        "--s3.bucket_name",
        type=str,
        help="The name of the S3 bucket to log to.",
        default="sn25-folding-mainnet",
    )

    parser.add_argument(
        "--s3.miner_bucket_name",
        type=str,
        help="The name of the S3 bucket that miners will upload their files to.",
        default="miner-upload-sn25",
    )

    parser.add_argument(
        "--s3.region_name",
        type=str,
        help="The region of the S3 bucket.",
        default="nyc3",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--neuron.suppress_cmd_output",
        action="store_true",
        help="If set, we suppress the text output of terminal commands to reduce terminal clutter.",
        default=True,
    )

    parser.add_argument(
        "--neuron.max_workers",
        type=int,
        help="Total number of subprocess that the miner is designed to run.",
        default=8,
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="folding-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call. (seconds)",
        default=45,
    )

    parser.add_argument(
        "--neuron.update_interval",
        type=float,
        help="The interval in which the validators query the miners for updates. (seconds)",
        default=60,  # samples every 5-minutes in the simulation.
    )

    parser.add_argument(
        "--neuron.queue_size",
        type=int,
        help="The number of jobs to keep in the queue.",
        default=10,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=10,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.positive_alpha",
        type=float,
        help="Positive alpha parameter, how much to add of the new observation when the reward is positive.",
        default=0.10,
    )

    parser.add_argument(
        "--neuron.negative_alpha",
        type=float,
        help="Negative alpha parameter, how much to add of the new observation when the reward is 0.",
        default=0.03,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=20_000,
    )

    parser.add_argument(
        "--neuron.synthetic_job_interval",
        type=float,
        help="The amount of time that the synthetic job creation loop should wait before checking the queue size again.",
        default=60,
    )

    parser.add_argument(
        "--neuron.organic_enabled",
        action="store_true",
        help="Set this flag to enable organic scoring.",
        default=False,
    )

    parser.add_argument(
        "--neuron.organic_trigger",
        type=str,
        help="Organic query validation trigger mode (seconds or steps).",
        default="seconds",
    )

    parser.add_argument(
        "--neuron.organic_trigger_frequency",
        type=float,
        help="Organic query sampling frequency (seconds or steps value).",
        default=120.0,
    )

    parser.add_argument(
        "--neuron.organic_trigger_frequency_min",
        type=float,
        help="Minimum organic query sampling frequency (seconds or steps value).",
        default=5.0,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="folding-openmm",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="macrocosmos",
    )

    parser.add_argument(
        "--organic_whitelist",
        nargs="+",  # Accepts one or more values as a list
        help="The validator will only accept organic queries from a list of whitelisted hotkeys.",
        default=[
            "5Cg5QgjMfRqBC6bh8X4PDbQi7UzVRn9eyWXsB8gkyfppFPPy",
        ],
    )
    parser.add_argument(
        "--neuron.gjp_address",
        type=str,
        help="The IP address and port of the global job pool.",
        default="167.99.209.27:8030",
    )

    parser.add_argument(
        "--neuron.organic_api.port",
        type=int,
        help="The port of the organic API.",
        default=8031,
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    # logger.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
