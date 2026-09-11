import os
from pathlib import Path

HOME = Path.home()
DATA = Path(os.environ.get("MBENCH_HOME", HOME / ".local/share/mbench"))
CACHE = Path(os.environ.get("XDG_CACHE_HOME", HOME / ".cache")) / "mbench"
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "mbench"
DB = DATA / "bench.db"
RUNS = DATA / "runs"
BOARD = DATA / "leaderboard.html"
PROFILES = CONFIG / "models.toml"
SETTINGS = CONFIG / "config.toml"
HARDWARE = CONFIG / "hardware.json"
LLAMA_SWAP_CONFIG = Path(os.environ.get("MBENCH_SWAP_CONFIG", HOME / ".config/llama-swap/config.yaml"))
OMP_MODELS = Path(os.environ.get("MBENCH_OMP_MODELS", HOME / ".omp/agent/models.yml"))
SWAP_URL = os.environ.get("MBENCH_SWAP_URL", "http://127.0.0.1:8081").rstrip("/")
PACKAGE = Path(__file__).resolve().parent
ASSETS = PACKAGE / "assets"
