# configs/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Root of the project (one level up from configs/)
ROOT_DIR = Path(__file__).parent.parent

# Paths — works on both Windows and Linux
DATA_DIR           = ROOT_DIR / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR       = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODEL_DIR          = ROOT_DIR / os.getenv("MODEL_DIR", "models")

# Redis (we'll run Redis locally in Step 8)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Model hyperparameters — centralised here, not buried in model files
TWO_TOWER_EMBED_DIM     = 128
TWO_TOWER_HIDDEN_DIM    = 256
FAISS_TOP_K_CANDIDATES  = 500
RERANK_TOP_K            = 20
CACHE_TTL_SECONDS       = 300

# Derived counts — populated after preprocessing
N_USERS  = None   # set at training time
N_ITEMS  = None   # set at training time