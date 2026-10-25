import shutil
from pathlib import Path


def clear_folder(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
