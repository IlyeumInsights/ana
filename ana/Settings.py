"""
ANA"s settings, including paths, location and others.

Available constants:

- MAIN_DATASET_LOC
- TRAIN_DATASET_LOC
- TEST_DATASET_LOC
- MODEL_DEST
- MODEL_VECT_DEFAULT
- MODEL_PIPELINES
- KL_LOC
- KL_VOCAB
- KL_RULES
- AUTOML_TPOT
"""

# Location of the full dataset to load
MAIN_DATASET_LOC = f"datasets\\service_contract_sample"

# Location of the train dataset
TRAIN_DATASET_LOC = f"datasets\\service_contract_sample"

# Location of the dataset to evaluate pipelines
TEST_DATASET_LOC = f"datasets\\service_contract_sample"

# Annotation csv files names (ordered by granularity)
# DATASET_LABEL_FILES = ["clauseClasses.csv", "sentClasses.csv"]
DATASET_LABEL_FILES = ["clauses_label.csv", "sentences_label.csv"]

# Destination of exported/saved models
MODEL_DEST = f".\\models\\"

# Default destination, mainly used for testing and debug
MODEL_VECT_DEFAULT = MODEL_DEST+f"vectorizer.pickle"

# Default destination of mapping, mainly used for testing and debug
MODEL_MAP_DEFAULT = MODEL_DEST+f"mapping.pickle"

# Location of pipelines related models in models directory
MODEL_PIPELINES = MODEL_DEST+f"pipelines\\"

# Knowledge location
KL_LOC = f"knowledge\\"

KL_VOCAB = KL_LOC+f"vocab\\"

KL_RULES = KL_LOC+f"rules\\policies.swrl"

# Activate usage of TPOT AutoML in the ensemble classifier AutoML process
AUTOML_TPOT = False
