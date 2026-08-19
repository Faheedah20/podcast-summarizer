"""Load local .env values without requiring an external dotenv package."""

import os
from pathlib import Path


def load_project_env(project_dir: Path) -> None:
    env_path = project_dir / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name:
            os.environ[name] = value
