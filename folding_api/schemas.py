import traceback
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field
from fastapi import Header
import time
from typing import Annotated
from substrateinterface import Keypair
from folding.utils.logger import logger
from hashlib import sha256


class FoldingParams(BaseModel):
    pdb_id: str
    source: Literal["rcsb", "pdbe"]
    ff: str
    water: str
    box: Literal["cube", "dodecahedron", "octahedron"]
    temperature: float
    friction: float
    epsilon: float
    time_to_live: Optional[int] = None
    pdb_file_path: Optional[str] = None


class FoldingSchema(BaseModel):
    """
    Represents a request to a validator.
    """

    pdb_id: str = Field(
        "1ubq", description="The PDB identifier for the selected response source."
    )
    source: Literal["rcsb", "pdbe"] = Field(
        "rcsb",
        description="The source of the folding request from two recognized databases.",
    )
    ff: str = Field(
        "charmm36.xml", description="The force field for the selected response source."
    )
    water: str = Field(
        "charmm36/water.xml",
        description="The water model for the selected response source.",
    )
    box: Literal["cube", "dodecahedron", "octahedron"] = Field(
        ..., description="The box type for the selected response source."
    )
    temperature: float = Field(
        ..., description="The temperature for the selected response source."
    )
    friction: float = Field(
        ..., description="The friction coefficient for the selected response source."
    )
    epsilon: float = Field(
        ...,
        description="The base epsilon that should be used for the challenge. Represented in %/100",
    )
    user_id: str = Field(
        ..., description="The user identifier for tracking job ownership."
    )

    validator_uids: list[int] = Field(
        ..., description="The validator identifier for the selected response source."
    )
    time_to_live: Optional[int] = Field(
        None, description="The time to live for the job in seconds."
    )
    num_validators_to_sample: Optional[int] = Field(
        None, description="The number of validators to sample."
    )

    timeout: int = Field(5, description="The time in seconds to wait for a response.")

    @property
    def folding_params(self) -> FoldingParams:
        """Get the simulation parameters for SimulationConfig"""
        return FoldingParams(
            pdb_id=self.pdb_id,
            source=self.source,
            ff=self.ff,
            water=self.water,
            box=self.box,
            temperature=self.temperature,
            friction=self.friction,
            epsilon=self.epsilon,
            time_to_live=self.time_to_live,
        )

    @property
    def api_parameters(self):
        """Get the API parameters for the Folding API"""
        return {
            "validator_uids": self.validator_uids,
            "num_validators_to_sample": self.num_validators_to_sample,
        }


class FoldingReturn(BaseModel):
    """
    Represents a response from a validator.
    """

    uids: List[int] = Field(..., description="The uids of the response.")
    hotkeys: List[str] = Field(..., description="The hotkeys of the response.")
    status_codes: List[Any | None] = Field(
        ..., description="The status code of the response."
    )
    job_id: List[str | None] = Field(..., description="The job id of the response.")


class EpistulaHeaders:
    def __init__(
        self,
        version: str = Header(..., alias="Epistula-Version", pattern="^2$"),
        timestamp: str = Header(default=str(time.time()), alias="Epistula-Timestamp"),
        uuid: str = Header(..., alias="Epistula-Uuid"),
        signed_by: str = Header(..., alias="Epistula-Signed-By"),
        request_signature: str = Header(..., alias="Epistula-Request-Signature"),
    ):
        self.version = version
        self.timestamp = timestamp
        self.uuid = uuid
        self.signed_by = signed_by
        self.request_signature = request_signature

    def verify_signature_v2(
        self, body: bytes, now: float
    ) -> Optional[Annotated[str, "Error Message"]]:
        try:
            if not isinstance(self.request_signature, str):
                raise ValueError("Invalid Signature")

            timestamp = int(float(self.timestamp))
            if not isinstance(timestamp, int):
                raise ValueError("Invalid Timestamp")

            if not isinstance(self.signed_by, str):
                raise ValueError("Invalid Sender key")

            if not isinstance(self.uuid, str):
                raise ValueError("Invalid uuid")

            if not isinstance(body, bytes):
                raise ValueError("Body is not of type bytes")

            ALLOWED_DELTA_MS = 8000
            keypair = Keypair(ss58_address=self.signed_by)

            if timestamp + ALLOWED_DELTA_MS < now:
                raise ValueError("Request is too stale")

            message = f"{sha256(body).hexdigest()}.{self.uuid}.{self.timestamp}."
            logger.debug("verifying_signature", message=message)

            verified = keypair.verify(message, self.request_signature)
            if not verified:
                raise ValueError("Signature Mismatch")

            return None

        except Exception as e:
            logger.error(f"signature_verification_failed: {traceback.format_exc()}")
            return str(e)


class APIKeyBase(BaseModel):
    owner: str
    rate_limit: str
    is_active: bool = True


class APIKey(APIKeyBase):
    key: str


class APIKeyCreate(BaseModel):
    owner: str
    rate_limit: str = "100/hour"


class APIKeyResponse(APIKeyBase):
    key: str


class PDB(BaseModel):
    pdb_id: str
    source: str


class PDBSearchResponse(BaseModel):
    """
    Represents a response from a PDB search.
    """

    matches: List[PDB] = Field(..., description="List of matching PDB IDs")
    total: int = Field(..., description="Total number of matches found")


class UserPDBResponse(BaseModel):
    """
    Represents a response containing PDB IDs for a specific user.
    """

    user_id: str = Field(..., description="The user identifier")
    pdb_ids: List[str] = Field(..., description="List of PDB IDs for the user")
    total: int = Field(..., description="Total number of PDB IDs found")


class PDBInfoResponse(BaseModel):
    """
    Represents detailed information about a PDB structure from RCSB.
    """

    pdb_id: str = Field(..., description="PDB ID")
    molecule_name: Optional[str] = Field(
        None, description="Name of the molecule/protein"
    )
    classification: Optional[str] = Field(None, description="Structural classification")
    organism: Optional[str] = Field(None, description="Source organism")
    expression_system: Optional[str] = Field(None, description="Expression system used")


class Job(BaseModel):
    id: str
    type: Literal["organic", "synthetic"]
    job_id: str
    pdb_id: str
    created_at: str
    status: Literal["active", "inactive", "failed"]
    priority: int
    validator_hotkey: str
    best_hotkey: str
    s3_links: dict[str, str]


class JobPoolResponse(BaseModel):
    """
    Represents a response from a job pool.
    """

    jobs: List[Job] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")


class Miner(BaseModel):
    uid: str
    hotkey: str
    energy: list[dict[str, float]]
    final_energy: float


class JobResponse(BaseModel):
    pdb_id: str
    pdb_data: str
    pdb_file_link: str
    organism: Optional[str] = Field(None, description="Source organism")
    classification: Optional[str] = Field(None, description="Structural classification")
    expression_system: Optional[str] = Field(None, description="Expression system used")
    mutations: Optional[bool] = Field(None, description="Mutations in the PDB")
    source: str
    temperature: float
    friction: float
    pressure: float
    time_to_live: float
    ff: str
    water: str
    box: str
    miners: List[Miner]
    status: Literal["active", "inactive", "failed"] = Field(
        "inactive", description="Job status"
    )
    created_at: str = Field("", description="Job creation timestamp")
    updated_at: str = Field("", description="Job last update timestamp")
