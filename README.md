# Medical Artificial Intelligence

This repository contains two course/project deliverables:

- `TASK1-API/`: PDF-based medical information extraction with an LLM API.
- `TASK2/`: Heart failure mortality prediction, risk factor analysis, and paper materials.

## Repository Structure

### Task 1

Path: `TASK1-API/`

Core files:

- `TASK1-API/main.py`: main script for PDF parsing, prompt construction, API calling, and JSON export.
- `TASK1-API/output.json`: example structured output.
- `TASK1-API/A case of portal vein recanalization and symptomatic heart failure.pdf`: input case PDF.

### Task 2

Path: `TASK2/`

Core files:

- `TASK2/classification_prediction.py`: model benchmarking and best-model artifact generation.
- `TASK2/new_patient_prediction.py`: mortality probability prediction for a new patient.
- `TASK2/risk_factor_detection.py`: risk factor detection and feature importance analysis.
- `TASK2/eda_processing.py`: preprocessing and descriptive analysis.
- `TASK2/requirements.txt`: Python dependencies for Task 2.
- `TASK2/artifacts/best_mortality_predictor.joblib`: saved best-performing predictor.
- `TASK2/results/`: experiment outputs and prediction results.
- `TASK2/paper/`: paper source files, figures, tables, and bibliography.

## How To Run

### Task 1

Environment requirements:

- Python 3.9+
- A valid DashScope API key in environment variable `DASHSCOPE_API_KEY`

Suggested dependencies:

```bash
pip install requests python-dotenv PyPDF2
```

Run:

```bash
cd TASK1-API
python main.py
```

Output:

- `TASK1-API/output.json`

### Task 2

Environment requirements:

- Python 3.9+

Install dependencies:

```bash
cd TASK2
pip install -r requirements.txt
```

Run the main experiments:

```bash
python classification_prediction.py
python risk_factor_detection.py
python eda_processing.py
```

Run new-patient mortality probability prediction:

```bash
python new_patient_prediction.py
```

Or use a CSV input:

```bash
python new_patient_prediction.py --input new_patient_example.csv --output results/new_patient_prediction_result.csv
```

## Main Deliverables

### Task 1 Deliverables

- Structured medical entity extraction from a clinical PDF
- JSON output file
- Reproducible API-based extraction script

### Task 2 Deliverables

- Risk factor analysis results
- Model benchmark and preprocessing sensitivity analysis
- Best-model artifact for patient-level inference
- New-patient mortality probability prediction
- Full paper source under `TASK2/paper/`

