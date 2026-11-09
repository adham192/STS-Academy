"""Run the API locally with auto-reload:  uv run python scripts/run_dev.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    # log_config=None -> uvicorn keeps OUR logging config instead of its own
    uvicorn.run(
        "app.main:app",
        app_dir=str(ROOT),
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT / "app")],
        log_config=None,
    )