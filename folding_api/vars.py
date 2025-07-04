import argparse
import bittensor as bt
import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address

# from folding_api.auth import APIKeyManager
from folding_api.chain import SubtensorService
from atom.epistula.epistula import Epistula

parser = argparse.ArgumentParser()

# Add arguments
bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
parser.add_argument("--netuid", type=int, help="Subnet netuid", default=25)
parser.add_argument(
    "--api-key-file", type=str, help="API key file", default="api_keys.json"
)
parser.add_argument(
    "--gjp-address", type=str, help="GJP API address", default="167.99.209.27:4001"
)

bt_config = bt.config(parser)

logger = structlog.get_logger()
# Add rate limiting
limiter = Limiter(key_func=get_remote_address)

# api_key_manager = APIKeyManager()
subtensor_service = SubtensorService(
    config=bt_config
)  # Pass config to SubtensorService

epistula = Epistula()
