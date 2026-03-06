#!/usr/bin/env python3
"""Execute a notebook with papermill and save the executed version."""

import sys
import time
from pathlib import Path

import papermill as pm


def execute_notebook(notebook_path: str, output_base: str = "notebooks") -> None:
    notebook_path = notebook_path.removeprefix("notebooks/")
    nb_file = Path(output_base) / notebook_path

    if not nb_file.exists():
        print(f"Error: Notebook not found: {nb_file}", file=sys.stderr)
        sys.exit(1)

    nb_dir = nb_file.parent
    output_dir = nb_dir / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    output_nb = nb_dir / f"{nb_file.stem}-executed.ipynb"

    print(f"Executing notebook: {nb_file}")
    print(f"Output notebook: {output_nb}")

    start = time.monotonic()
    try:
        pm.execute_notebook(
            str(nb_file),
            str(output_nb),
            kernel_name="python3",
            log_output=True,
        )
        duration = time.monotonic() - start
        print(f"\nNotebook executed successfully: {output_nb}")
        print(f"  Duration: {duration:.1f}s")
    except Exception as e:
        print(f"\nNotebook execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: execute_notebook.py <notebook_path>", file=sys.stderr)
        sys.exit(1)
    execute_notebook(sys.argv[1])
