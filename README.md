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

### Step 5: Verify Installation
Check that all required packages are installed:
```bash
pip3 list
```

## Usage

### Command-Line Interface (CLI)
Run the pipeline using the `pipeline.py` script:

#### Training the Model
```bash
python pipeline.py train <path_to_csv> --epochs <num_epochs> --batch <batch_size>
```
- `<path_to_csv>`: Path to the CSV file containing timestamped adjusted closing prices.
- `--epochs`: (Optional) Number of training epochs. Default is 30.
- `--batch`: (Optional) Batch size for training. Default is 64.
- You can also edit the epochs, batch size, etc directly in the pipeline.py file, there
  it will also tell you what adjusting them does and how it affects the resulting model

#### Example
```bash
python pipeline.py train data.csv --epochs 40 --batch 32
```

#### Making Predictions
```bash
python pipeline.py infer <price_1> <price_2> ... <price_30>
```
- Provide exactly 30 recent adjusted closing prices (oldest to newest).

#### Example
```bash
python pipeline.py infer 101.2 101.4 101.1 101.8 102.0 102.2 102.4 102.1 102.5 102.9 \
103.1 103.4 103.2 103.8 104.0 104.3 104.5 104.8 105.0 105.2 \
105.5 105.7 105.8 106.0 106.3 106.5 106.7 107.0 107.2 107.4
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
- The trained model is saved in the `model_artifact` directory as `lstm.h5`.
- Metadata about the training process is saved as `meta.pkl` in the same directory.
- The script prints the predicted next-bar return percentage for a sample input.