# Househunter Case

The Househunter case is part of the ExquAIro course on Fundamental Modelling Techniques. 

## Repository Structure

- `script_househunter.ipynb`: Main notebook containing the complete analysis and modeling.
- `190322 - HouseTable_vDef_excel.xlsx`: Dataset containing historical house sales data.
- `config.yaml`: Configuration file with working directory settings.
- `pyproject.toml`: Project configuration with dependencies.
- `Scripts/`: Directory containing utility functions:
  - `__init__.py`: Package initialization.
  - `formatting_functions.py`: Functions for results formatting.
  - `modeling_functions.py`: Functions for building and training models.
  - `performance_metrics_functions.py`: Functions for evaluating model performance.
  - `plotting_functions.py`: Functions for creating visualizations.

## Installation and Setup

1. Ensure you have Python 3.13.9 installed.
2. Install `uv` for dependency management if not already installed.
3. Navigate to the Househunter case directory.
4. Run `uv sync` to create a virtual environment and install dependencies.
5. Open the project in VSCode or your preferred IDE.
6. For Jupyter notebooks, select the virtual environment as the kernel.

## Data

The dataset (`190322 - HouseTable_vDef_excel.xlsx`) contains historical house sales data with various features such as location, size, number of rooms, and sale prices.

## Usage

1. Run `script_househunter.ipynb` for the full analysis.

## Notes

- Ensure the working directory in `config.yaml` matches your local path.
