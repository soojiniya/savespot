from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebook" / "16_TAAS_전국_사고특성_EDA.ipynb"
RUNTIME_ROOT = PROJECT_ROOT / "work" / "jupyter_runtime"


def main() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    for name, child in {
        "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "jupyter_config",
        "JUPYTER_DATA_DIR": "jupyter_data",
        "JUPYTER_RUNTIME_DIR": "jupyter_runtime",
    }.items():
        path = RUNTIME_ROOT / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / "notebook")}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
