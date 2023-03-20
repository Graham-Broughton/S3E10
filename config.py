import os
from dataclasses import dataclass

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
RAW_DATA = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA = os.path.join(DATA_PATH, "processed")


@dataclass
class CFG:
    NROWS: int = 100
    NCOLS: int = 30
    BASE_PATH: str = BASE_PATH
    DATA_PATH: str = DATA_PATH
    RAW_DATA: str = RAW_DATA
    PROCESSED_DATA: str = PROCESSED_DATA
    SEED: int = 69
    NFOLDS: int = 3
    REPEATS: int = 3
    XG_PATIENCE: int = 50
    CB_PATIENCE: int = 100
    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    LR: int = 1e-3
