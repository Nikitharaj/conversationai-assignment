#!/usr/bin/env python3
"""
Script to run Jupyter notebooks using the jupyter nbconvert command.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_notebook(notebook_path):
    """
    Run a Jupyter notebook using the jupyter nbconvert command.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if the notebook was executed successfully, False otherwise
    """
    print(f"Running notebook: {notebook_path}")

    # Create output path
    notebook_path = Path(notebook_path)
    output_dir = notebook_path.parent
    output_name = f"{notebook_path.stem}_executed{notebook_path.suffix}"
    output_path = output_dir / output_name

    print(f"Output will be saved to: {output_path}")

    # Build the command
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        str(notebook_path),
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
    ]

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Notebook executed successfully: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing notebook: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_notebooks.py <notebook_path> [<notebook_path> ...]")
        sys.exit(1)

    notebook_paths = sys.argv[1:]
    success_count = 0

    for notebook_path in notebook_paths:
        if run_notebook(notebook_path):
            success_count += 1

    total = len(notebook_paths)
    print(
        f"\nExecution summary: {success_count}/{total} notebooks executed successfully"
    )

    if success_count == total:
        print("All notebooks executed successfully!")
        sys.exit(0)
    else:
        print(f"Failed to execute {total - success_count} notebooks")
        sys.exit(1)


if __name__ == "__main__":
    main()
