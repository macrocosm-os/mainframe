import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from folding_api.chain import SubtensorService
from folding_api.protein import router
from folding_api.utility_endpoints import router as utility_router
from folding_api.validator_registry import ValidatorRegistry
from folding_api.auth import APIKeyManager, get_api_key, api_key_router
from folding_api.database import db_manager
from folding_api.vars import (
    bt_config,
    limiter,
    logger,
    subtensor_service,
)


async def sync_metagraph_periodic(
    subtensor_service: SubtensorService, validator_registry: ValidatorRegistry
):
    """Background task to sync metagraph every hour"""
    while True:
        await asyncio.sleep(600)
        try:
            logger.info("Syncing metagraph")
            # Run the synchronous function in a thread pool
            await asyncio.to_thread(subtensor_service.resync_metagraph)
            await asyncio.to_thread(validator_registry.update_validators)
            logger.info("Metagraph sync completed")
        except Exception as e:
            logger.error(f"Error syncing metagraph: {e}")


# Initialize API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    logger.info("initializing_services")
    logger.info(f"Using config: {bt_config}")

    validator_registry = ValidatorRegistry()
    app.state.validator_registry = validator_registry

    # Initialize API key manager
    api_key_manager = APIKeyManager()
    app.state.api_key_manager = api_key_manager

    # Initialize database
    await db_manager.init_database()
    app.state.db_manager = db_manager

    # Start background sync task
    app.state.sync_task = asyncio.create_task(
        sync_metagraph_periodic(subtensor_service, validator_registry)
    )
    logger.info("Started background sync task")

    yield

    # Cleanup
    logger.info("cleaning_up_services")
    if app.state.sync_task:
        logger.info("Canceling background sync task")
        app.state.sync_task.cancel()
        try:
            await app.state.sync_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan, name="folding_api", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)

# Add API key dependency to all routes
app.dependency_overrides[get_api_key] = get_api_key

# Include routes
app.include_router(router, dependencies=[Depends(get_api_key)])
app.include_router(api_key_router)  # API key management routes
app.include_router(utility_router)  # Utility endpoints
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8029)
