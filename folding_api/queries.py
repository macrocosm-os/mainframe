from fastapi import HTTPException, UploadFile, File
from typing import Optional
from collections import defaultdict

from folding_api.schemas import FoldingSchema, FoldingReturn
from folding_api.utils import make_request
from folding_api.validator_registry import ValidatorRegistry
from folding_api.database import db_manager


async def query_validators(
    schema: FoldingSchema,
    validator_registry: ValidatorRegistry,
    pdb_file: Optional[UploadFile] = File(None),
) -> FoldingReturn:
    """
    Query validators with the given parameters and return a streaming
    """
    # Get the UIDs (and axons) to query by looking at the top validators
    # check if uids are in registry, if all of them are not default to random choice
    if schema.api_parameters["validator_uids"] is not None and all(
        uid in validator_registry.validators
        for uid in schema.api_parameters["validator_uids"]
    ):
        uids = schema.api_parameters["validator_uids"]
        validators = {uid: validator_registry.validators[uid] for uid in uids}
    else:
        validators = validator_registry.get_available_axons(
            k=schema.api_parameters["num_validators_to_sample"]
        )

    if validators is None:
        raise HTTPException(status_code=404, detail="No validators available")

    validator_responses = []
    validator_uids = []
    for uid, validator in validators.items():
        validator_responses.append(
            await make_request(validator.address, schema.folding_params, pdb_file)
        )
        validator_uids.append(uid)

    response_information = defaultdict(list)
    if len(validator_responses) == 0:
        raise HTTPException(status_code=404, detail="No validators available")
    for resp, uid in zip(validator_responses, validator_uids):
        response_information["hotkeys"].append(validators[uid].hotkey)
        response_information["uids"].append(uid)
        if resp is not None:
            response_information["status_codes"].append(resp.status_code)
            if resp.status_code == 200:
                job_id = resp.json()["job_id"]
                response_information["job_id"].append(job_id)
                
                # Insert successful job into database
                try:
                    await db_manager.insert_protein_job(
                        job_id=job_id,
                        pdb_id=schema.pdb_id,
                        user_id=schema.user_id
                    )
                except Exception as e:
                    # Log the error but don't fail the request
                    from loguru import logger
                    logger.error(f"Failed to insert protein job to database: {e}")
            else:
                response_information["job_id"].append(None)
        else:
            response_information["status_codes"].append(None)
            response_information["job_id"].append(None)

    return FoldingReturn(**response_information)
