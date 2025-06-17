# Simulation constants
MIN_LOGGING_ENTRIES = 500
MIN_SIMULATION_STEPS = 1000
MAX_SIMULATION_STEPS_FOR_EVALUATION = 3000
ANOMALY_THRESHOLD = 1  # The percentage that we allow the energy to differ from the miner to the validator.

# Evaluation constants
XML_CHECKPOINT_THRESHOLD = 5  # Percent
GRADIENT_THRESHOLD_INTERMEDIATE = 100  # kJ/mol/nm
GRADIENT_THRESHOLD_FINAL = 35  # kJ/mol/nm
GRADIENT_WINDOW_SIZE = 40  # Number of steps to calculate the gradients over.
ENERGY_WINDOW_SIZE = (
    10  # Number of steps to compute median/mean energies when comparing
)
MINER_CHECKPOINT_SIMILARITY_TOLERANCE = (
    0.05  # Tolerance for cpts to be considered similar. NOT in percent.
)

# MinerRegistry constants
MAX_JOBS_IN_MEMORY = 1000
STARTING_CREDIBILITY = 0.50

# Reward Constants
TOP_SYNTHETIC_MD_REWARD = 0.80
DIFFERENCE_THRESHOLD = 1e-6  # The threshold for 2 energy values to be considered equal
CREDIBILITY_ALPHA_POSITIVE = 0.15
CREDIBILITY_ALPHA_NEGATIVE = 0.25

# Validation constants
MAX_CHECKPOINTS_TO_VALIDATE = 3
INTERMEDIATE_CHECKPOINT_STEPS = 1000
