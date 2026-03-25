# Welcome to organelle-morphology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/organelle-morphology/ci.yml?branch=main)](https://github.com/ssciwr/organelle-morphology/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/organelle-morphology/badge/)](https://organelle-morphology.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/organelle-morphology/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/organelle-morphology)

**This project is currently under construction**

## Installation
To get started, clone the repository and set up the Conda environment:

```bash
git clone https://github.com/ssciwr/organelle-morphology.git
cd organelle-morphology
conda env create -f environment-dev.yml
conda activate morph
```

### Interactive UI
To start the graphical user interface, run:

```bash
marimo run notebooks/ui.py
```
Click on the displayed link or navigate to http://localhost:2718 in your web browser.

### Tests
The test suite can be run using pytest:
```bash
python -m pytest
```
