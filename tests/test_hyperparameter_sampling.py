import pytest

from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
from folding.validators.md.hyperparameters import HyperParameters

BOX = ["cubic", "dodecahedron", "octahedron"]


def test_total_combinations():
    hp = HyperParameters()

    expected_number_of_combinations = 0
    for field in FORCEFIELD_REGISTRY.values():
        FF = field().forcefields
        WATER = field().waters

        expected_number_of_combinations += (
            len(FF) * len(WATER) * len(BOX) + 1
        )  # +1 for the recommended configuration

    assert hp.TOTAL_COMBINATIONS == expected_number_of_combinations


def test_exclude_parameter():
    FIELD_NAME = "amber14"
    WATER_FILE = "FAKE_WATER.xml"
    hp = HyperParameters(exclude={"FF": "amber14/protein.ff14SB.xml"})

    field = FORCEFIELD_REGISTRY[FIELD_NAME]()

    assert hp.TOTAL_COMBINATIONS == len(field.waters) * len(BOX)

    # Test with a water that does not exist in the possible set.
    with pytest.raises(ValueError, match=f"Selected water {WATER_FILE} not found"):
        HyperParameters(
            exclude={"FF": "amber14/protein.ff14SB.xml", "WATER": WATER_FILE}
        )
