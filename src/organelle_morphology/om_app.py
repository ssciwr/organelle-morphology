"""
Script to run the marimo application for organelle morphology analysis.
This script executes 'marimo run src/app/ui.py' in the current environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_marimo_app():
    """Run the marimo application."""
    ui_path = Path(__file__).parent.parent / "app" / "ui.py"

    if not ui_path.exists():
        print(f"Error: ui.py not found at {ui_path}")
        sys.exit(1)

    venv_bin = str(Path(sys.executable).parent)
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    env["PATH"] = venv_bin + os.pathsep + current_path

    cmd = ["marimo", "run", str(ui_path)]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,  # capture_output=True, text=True,
        )
        print("Marimo app started successfully!")
        print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running marimo app: {e}")
        print(f"stderr: {e.stderr}")
        return e.returncode
    except FileNotFoundError:
        print("Error: marimo command not found. Please ensure marimo is installed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_marimo_app())
