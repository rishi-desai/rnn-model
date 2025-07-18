# LSTM Financial Time Series Forecasting Pipeline

This repository contains a plug-and-play pipeline for forecasting financial time series using a lightweight two-layer LSTM model. The pipeline is designed to be user-friendly and requires minimal coding or machine learning knowledge.

## Features
- Ingests a CSV file of timestamped adjusted closing prices.
- Automatically windows the series into overlapping 30-bar segments.
- Computes each segment's next-bar return.
- Trains a lightweight two-layer LSTM model.
- Saves the trained model as a TensorFlow artifact.
- Predicts the next-bar return percentage for a given 30-bar input.

## Requirements
- Python 3.8+
- Required Python packages: `pandas`, `numpy`, `tensorflow`

## Setup Instructions for a New Computer

### Step 1: Install Homebrew (macOS Only)
If you are on macOS, install Homebrew, a package manager, to simplify the setup process:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python Using Homebrew (macOS Only)
Install Python 3.8+ using Homebrew:
```bash
brew install python@3.10
```

### Step 3: Install Virtual Environment Tool
Create a virtual environment to isolate dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### Step 4: Install Required Packages
Install the required Python packages using:
```bash
pip3 install -r requirements.txt
```

### Step 6: Verify Installation
Check that all required packages are installed:
```bash
pip3 list
```

## Usage

### Command-Line Interface (CLI)
Run the pipeline using the `run_pipeline.py` script:
```bash
python run_pipeline.py --csv <path_to_csv> --model_path <path_to_save_model>
```

- `--csv`: Path to the CSV file containing timestamped adjusted closing prices.
- `--model_path`: (Optional) Path to save or load the trained model. Default is `lstm_model`.

### Example
```bash
python run_pipeline.py --csv data.csv
```

## Input CSV Format
The input CSV file should have the following columns:
- `timestamp`: The timestamp of the data point.
- `adjusted_close`: The adjusted closing price.

Example:
```csv
timestamp,adjusted_close
2025-01-01,100.0
2025-01-02,101.5
2025-01-03,102.0
```

## Output
- The trained model is saved as a TensorFlow artifact.
- The script prints the predicted next-bar return percentage for a sample input.

## Optional GUI (Coming Soon)
A graphical user interface (GUI) will be added to make the pipeline even more accessible.

## License
This project is licensed under the MIT License.