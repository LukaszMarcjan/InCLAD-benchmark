from datetime import datetime
from pathlib import Path


def default_results_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results_{timestamp}")
