# Welcome to organelle-morphology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/organelle-morphology/ci.yml?branch=main)](https://github.com/ssciwr/organelle-morphology/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/organelle-morphology/badge/)](https://organelle-morphology.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/organelle-morphology/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/organelle-morphology)

**This project is currently under construction**

To start the user interface under Linux:
```bash
   git clone [https://github.com/ssciwr/organelle-morphology.git](https://github.com/ssciwr/organelle-morphology.git)
   cd organelle-morphology
   conda env create -f environment-dev.yml
   conda activate morph
   pip install -e .
   cd notebooks
   marimo run ui.py --host 0.0.0.0 --no-token
```
Navigate to http://localhost:2718
