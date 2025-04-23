import uvicorn
from fastapi import FastAPI
from folding.base import validator
from folding.organic.organic import router as organic_router
from folding.utils.logging import logger
import multiprocessing
from multiprocessing.connection import Connection
import pickle
from typing import Optional

app = FastAPI()

app.include_router(organic_router)


async def start_organic_api(organic_validator, config):
    app.state.validator = organic_validator
    app.state.config = config

    logger.info(
        f"Starting organic API on  http://0.0.0.0:{config.neuron.organic_api.port}"
    )
    config = uvicorn.Config(
        "folding.organic.api:app",
        host="0.0.0.0",
        port=config.neuron.organic_api.port,
        loop="asyncio",
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


def api_process_main(pipe_connection: Connection, config):
    """
    Main function to run in the separate API process.
    Receives jobs from the API and sends them back to the main process.

    Args:
        pipe_connection: Connection to communicate with the main process
        config: Configuration for the API
    """
    from folding.organic.api import app
    import uvicorn
    from atom.organic_scoring.organic_queue import OrganicQueue
    import asyncio
    from asyncio import Task

    # Create a dummy validator object that will send jobs through the pipe
    class PipeOrganicValidator:
        def __init__(self, pipe_connection):
            self._organic_queue = OrganicQueue()
            self._pipe_connection = pipe_connection
            self._check_queue_task: Optional[Task] = None

        async def check_queue(self):
            """Periodically check the queue and send jobs to the main process"""
            while True:
                try:
                    if not self._organic_queue.is_empty():
                        # Get all items from the queue
                        items = []
                        while not self._organic_queue.is_empty():
                            item = self._organic_queue.sample()
                            if item:
                                items.append(item)

                        # Send items through the pipe
                        if items:
                            logger.info(f"Sending {len(items)} jobs to main process")
                            self._pipe_connection.send(pickle.dumps(items))
                except Exception as e:
                    logger.error(f"Error checking queue: {e}")
                await asyncio.sleep(
                    1
                )  # Check more frequently than the main process reads

    # Set up the API
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create the validator
    organic_validator = PipeOrganicValidator(pipe_connection)

    # Start the queue checking task
    organic_validator._check_queue_task = loop.create_task(
        organic_validator.check_queue()
    )

    # Set up the app state
    app.state.validator = organic_validator
    app.state.config = config

    # Start the API
    uvicorn_config = uvicorn.Config(
        "folding.organic.api:app",
        host="0.0.0.0",
        port=config.neuron.organic_api.port,
        loop="asyncio",
        reload=False,
    )

    server = uvicorn.Server(uvicorn_config)
    loop.run_until_complete(server.serve())


def start_organic_api_in_process(config):
    """
    Start the organic API in a separate process and return a pipe connection
    to receive jobs from it.

    Args:
        config: Configuration for the API

    Returns:
        Connection: Pipe connection to receive jobs from the API process
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(
        target=api_process_main, args=(child_conn, config), daemon=True
    )
    process.start()
    logger.info(f"Started organic API in separate process (PID: {process.pid})")
    return parent_conn, process
