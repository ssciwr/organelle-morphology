import os
from pathlib import Path

import sys
from unittest.mock import patch
from organelle_morphology.cli.hpc_init import main


def test_script_files_exist():
    """Test that the required script files exist in the scripts directory."""

    scripts_dir = Path(__file__).parent.parent / "src" / "scripts"

    hpc_example = scripts_dir / "hpc_example.py"
    assert hpc_example.exists(), f"Expected {hpc_example} to exist"

    for choice in ["benchmark", "multi", "single"]:
        helix_run_file = scripts_dir / f"helix_run_{choice}.sh"
        assert helix_run_file.exists(), f"Expected {helix_run_file} to exist"


def test_main_function(tmp_path):
    """Test that the main function copies files correctly using tmp_path."""
    # Change to the temporary directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Test each choice
        for choice in ["benchmark", "multi", "single"]:
            # Mock sys.argv to simulate command-line arguments
            with patch.object(sys, "argv", ["hpc_init", choice]):
                main()

                # Check that the files were copied
                hpc_example_path = tmp_path / "hpc_example.py"
                helix_run_path = tmp_path / f"helix_run_{choice}.sh"

                assert hpc_example_path.exists(), (
                    f"hpc_example.py was not copied for choice '{choice}'"
                )
                assert helix_run_path.exists(), (
                    f"helix_run_{choice}.sh was not copied for choice '{choice}'"
                )

                # Verify content exists in files
                assert hpc_example_path.read_text() != "", (
                    f"hpc_example.py is empty for choice '{choice}'"
                )
                assert helix_run_path.read_text() != "", (
                    f"helix_run_{choice}.sh is empty for choice '{choice}'"
                )
    finally:
        os.chdir(original_cwd)
