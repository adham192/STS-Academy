"""Central logging configuration.

Call setup_logging(<folder>) once at the top of each entry file.
Calling it again later is harmless: basicConfig does nothing if logging
is already configured (important because Streamlit re-runs app.py on every click).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Constants used in every project: one name, one level, one format.
LOG_FILENAME = "app.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries emit their own DEBUG lines (every HTTP call etc.); keep them quiet.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "chromadb", "google_genai", "asyncio")


def _script_folder() -> Path:
    """Folder of the script being run (used when no folder is passed)."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(main_file).resolve().parent if main_file else Path.cwd()


def setup_logging(log_dir: Optional[Path] = None) -> Path:
    """Configure the root logger; returns the path of the log file in use."""
    log_file = (log_dir or _script_folder()) / LOG_FILENAME
    logging.basicConfig(
        filename=log_file,      # where the entries are written
        filemode="a",           # append, so older runs are kept
        level=LOG_LEVEL,        # DEBUG = capture everything
        format=LOG_FORMAT,      # timestamp | level | logger name | message
        datefmt=DATE_FORMAT,    # readable timestamp: 2026-10-31 19:48:05
        encoding="utf-8",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file