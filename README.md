# Pharmacy Billing vs. Ordering Reconciliation

A desktop-based analytics application for reconciling pharmacy medication billing and ordering data, identifying quantity discrepancies, and reducing manual reconciliation work.

## Overview

Pharmacy operations can experience differences between medication quantities billed and quantities ordered. Manually identifying these discrepancies across large datasets can be time-consuming and difficult to review consistently.

This project provides a local desktop workflow for processing billing and ordering datasets, aggregating medication quantities, and identifying shortages and overages.

## Key Features

- Processes pharmacy billing and ordering datasets
- Aggregates medication quantities across records
- Compares billed quantities with ordered quantities
- Identifies shortages and overages
- Presents reconciliation results through a desktop interface
- Supports analysis of large medication datasets
- Performs data processing locally on the user's machine

## Workflow

```text
Billing Data ──┐
               ├──> Data Processing
Ordering Data ─┘
                       │
                       ▼
              Quantity Aggregation
                       │
                       ▼
             Billing vs. Ordering
                 Reconciliation
                       │
                       ▼
             Shortage / Overage
                  Detection
                       │
                       ▼
                Desktop Results
````

## Technology
- Python
- PySide6
- pandas
- Jupyter Notebook

See `requirements.txt` for the full list of project dependencies.

## Repository Structure

```text
Billing-vs-Ordering-Reconciliation/
│
├── app.py
├── main_window.py
├── data_processing.py
├── medication_analysis.ipynb
├── requirements.txt
├── BillingOrderingDashboard.spec
└── README.md
```

### Main Components

**`app.py`**  
Application entry point.

**`main_window.py`**  
Desktop application interface.

**`data_processing.py`**  
Data processing and reconciliation logic.

**`medication_analysis.ipynb`**  
Exploratory medication data analysis.

**`BillingOrderingDashboard.spec`**  
Application build configuration.

## Scale Testing

The application was tested using synthetic datasets containing more than 700 medications to evaluate its ability to support larger reconciliation workflows.

Synthetic data is used for testing and demonstration purposes.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/suuhaiillkhann/Billing-vs-Ordering-Reconciliation.git
cd Billing-vs-Ordering-Reconciliation
```

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Data & Privacy

This public repository is intended for demonstration and portfolio purposes.

No real patient data, protected health information (PHI), pharmacy credentials, or proprietary production datasets are included in this repository.

Testing and demonstrations use synthetic data.

## Project Scope

This project focuses on descriptive reconciliation of pharmacy billing and ordering activity to identify quantity discrepancies such as shortages and overages.

It is an analytics and operational reconciliation tool, not a clinical decision-support system or an automated medication-ordering system.

Predictive medication-demand forecasting and production machine-learning workflows are outside the scope of this repository.

## Author

**Suhail**

Data Scientist | Applied Machine Learning · Predictive Modeling · Forecasting · MLOps

[Portfolio](https://www.suhailkhan.dev)
