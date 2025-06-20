from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not PROJECT_ROOT.exists():
    PROJECT_ROOT.mkdir()
assert PROJECT_ROOT.exists(), f"{PROJECT_ROOT.as_posix()} does not exist"

DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()
assert DATA_DIR.exists(), f"{DATA_DIR.as_posix()} does not exist"

CONF_DIR = PROJECT_ROOT / "conf"
if not CONF_DIR.exists():
    CONF_DIR.mkdir()
assert CONF_DIR.exists(), f"{CONF_DIR.as_posix()} does not exist"

RUNS_DIR = PROJECT_ROOT / "runs"
if not RUNS_DIR.exists():
    RUNS_DIR.mkdir()
assert RUNS_DIR.exists(), f"{RUNS_DIR.as_posix()} does not exist"