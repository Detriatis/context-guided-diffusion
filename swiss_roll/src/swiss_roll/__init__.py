from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert PROJECT_ROOT.exists(), f"{PROJECT_ROOT.as_posix()} does not exist"

DATA_DIR = PROJECT_ROOT / "data"
assert DATA_DIR.exists(), f"{DATA_DIR.as_posix()} does not exist"

CONF_DIR = PROJECT_ROOT / "conf"
assert CONF_DIR.exists(), f"{CONF_DIR.as_posix()} does not exist"

RUNS_DIR = PROJECT_ROOT / "runs"
assert RUNS_DIR.exists(), f"{RUNS_DIR.as_posix()} does not exist"