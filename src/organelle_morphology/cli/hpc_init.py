import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy example scripts from the scripts directory to current working directory"
    )
    parser.add_argument(
        "choice",
        choices=["benchmark", "multi", "single"],
        help="Choice of script to copy: benchmark, multi, or single",
    )

    args = parser.parse_args()

    # Define the base directory for scripts (relative to this file)
    script_dir = Path(__file__).parent.parent.parent / "scripts"

    # Define the files to copy
    hpc_example_file = script_dir / "hpc_example.py"
    helix_run_file = script_dir / f"helix_run_{args.choice}.sh"

    # Check if files exist
    if not hpc_example_file.exists():
        print(f"Error: {hpc_example_file} not found", file=sys.stderr)
        sys.exit(1)

    if not helix_run_file.exists():
        print(f"Error: {helix_run_file} not found", file=sys.stderr)
        sys.exit(1)

    # Copy the files to current working directory
    try:
        # Copy hpc_example.py
        shutil.copy(hpc_example_file, Path.cwd() / "hpc_example.py")
        print(f"Copied {hpc_example_file.name} to {Path.cwd()}")

        # Copy the chosen helix_run file
        shutil.copy(helix_run_file, Path.cwd() / f"helix_run_{args.choice}.sh")
        print(f"Copied {helix_run_file.name} to {Path.cwd()}")

        print("Files copied successfully!")

    except Exception as e:
        print(f"Error copying files: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
