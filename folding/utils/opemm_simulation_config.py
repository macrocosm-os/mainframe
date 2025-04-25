from openmm import app
import openmm as mm
from enum import Enum
from pydantic import BaseModel
from typing import Literal, Optional


class NonbondedMethod(Enum):
    PME = mm.app.PME
    NoCutoff = mm.app.NoCutoff


class Constraints(Enum):
    Unconstricted = None
    HBonds = app.HBonds
    AllBonds = app.AllBonds
    HAngles = app.HAngles


class SimulationConfig(BaseModel):
    ff: str
    water: str
    box: Literal["cube", "dodecahedron", "octahedron"]
    seed: Optional[int] = None
    temperature: float = 300.0
    time_step_size: float = 0.002
    time_units: str = "picosecond"
    save_interval_checkpoint: int = 10000
    save_interval_trajectory: int = 100
    save_interval_log: int = 10
    box_padding: float = 1.0
    friction: float = 1.0
    nonbonded_method: str = "NoCutoff"
    constraints: str = "HBonds"
    cutoff: Optional[float] = 1.0
    pressure: float = 1.0
    emtol: float = 1000.0
    rvdw: float = 1.2
    rcoulomb: float = 1.2
    fourier_spacing: float = 0.15
    tau_p: float = 5.0
    ref_p: float = 1.0
    ref_t: float = 300.0
    gen_temp: float = 300.0

    max_steps_nvt: int = 50000
    nsteps_minimize: int = 100
    simulation_steps: dict = {"nvt": 50000, "npt": 75000, "md_0_1": 500000}

    def get_config(self) -> dict:
        attributes = self.dict()
        attributes["nonbonded_method"] = NonbondedMethod[self.nonbonded_method].value
        attributes["constraints"] = Constraints[self.constraints].value
        return attributes

    def to_dict(self):
        return self.dict()
