import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    reload = os.getenv("APP_RELOAD", "true").lower() == "true"
    uvicorn.run(
        "app.main:app",
        app_dir=str(ROOT),
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=reload,
        reload_dirs=[str(ROOT / "app")] if reload else None,
        log_config=None,
    )