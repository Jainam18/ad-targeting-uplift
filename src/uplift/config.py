"""Single source of truth for constants from docs/00_analysis_plan.md."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
SQL = ROOT / "sql"

RAW_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
RAW_CSV = DATA_RAW / "criteo-uplift-v2.1.csv"
PARQUET = DATA_PROCESSED / "criteo.parquet"

SEED = 42
ALPHA = 0.05
POWER = 0.80

FEATURES: tuple[str, ...] = tuple(f"f{i}" for i in range(12))
TREATMENT = "treatment"
EXPOSURE = "exposure"
PRIMARY_OUTCOME = "conversion"
SECONDARY_OUTCOME = "visit"
FLAGS: tuple[str, ...] = (TREATMENT, PRIMARY_OUTCOME, SECONDARY_OUTCOME, EXPOSURE)
COLUMNS: tuple[str, ...] = FEATURES + FLAGS

EXPECTED_ROWS = 13_979_592

INTENDED_TREATMENT_SHARE = 0.85
SRM_P_THRESHOLD = 0.001
SMD_MATERIAL = 0.1
SMD_MINOR = 0.02
BALANCE_AUC_FLAG = 0.55
BALANCE_CV_FOLDS = 5

COST_PER_EXPOSED = 0.10
VALUE_PER_CONVERSION = 25.0
COST_RANGE = (0.01, 2.00)
VALUE_RANGE = (5.0, 200.0)

SPLIT_FRACTIONS = {"train": 0.4, "val": 0.2, "test": 0.4}
CUPAC_FOLDS = 2

HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 500,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": SEED,
}
