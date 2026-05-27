# ML-Project

Repository dedicated to the development, training, and evaluation of predictive models based on Machine Learning techniques for Chronic Kidney Disease prediction.

---

# Project Structure

```text
ML-Project/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── kidney_disease.csv
│   ├── interim/
│   │   └── kidney_disease_interim.csv
│   └── processed/
│       └── kidney_disease_encoded.csv
│
├── notebooks/
│   ├── EDA_Feature_Distributions.ipynb
│   ├── EDA_Missing_Values_Analysis.ipynb
│   ├── Encode_Categorical_Values.ipynb
│   ├── Validate_Interim_Dataset.ipynb
│   ├── Spot_Checking_ModelComparison.ipynb
│   ├── Spot_Checking_Balancing.ipynb
│   ├── Spot_Checking_DimensionalityReduction.ipynb
│   ├── Optimize_RF_Hyperparameters.ipynb
│   ├── Evaluate_RF_Balancing.ipynb
│   ├── Evaluate_RF_DimensionalityReduction.ipynb
│   └── Explainability_CKD_Selected_Model.ipynb
│
├── src/
│   ├── perform_minimal_data_cleaning.py
│   ├── modeling_utils.py
│   ├── modeling_workflow.py
│   ├── visualization_utils.py
│   └── explainability/
│       ├── __init__.py
│       ├── base_explainer.py
│       ├── factory.py
│       ├── reporting.py
│       ├── run_explainability.py
│       ├── visualizations.py
│       └── explainers/
│           ├── __init__.py
│           ├── knn_explainer.py
│           ├── linear_explainer.py
│           ├── model_agnostic_explainer.py
│           ├── svm_explainer.py
│           └── tree_explainer.py
│
├── tests/
│   ├── conftest.py
│   ├── test_base_explainer.py
│   ├── test_explainer_factory.py
│   ├── test_knn_explainer.py
│   ├── test_reporting.py
│   └── test_visualizations.py
│
└── reports/
    └── paper/
        ├── README.md
        ├── ckd_prediction.tex
        ├── references.bib
        ├── sbc-template.sty
        ├── sbc.bst
        ├── figures/
        └── sections/
            ├── introduction.tex
            ├── methodology.tex
            ├── results.tex
            └── conclusion.tex
```

---

# Environment Setup

## 1. Clone the repository

```bash
git clone git@github.com:fagundesariel/ML-Project.git
cd ML-Project
```

---

## 2. Create a virtual environment

```bash
python3.10 -m venv .venv
```

---

## 3. Activate the environment

### Linux / Mac:

```bash
source .venv/bin/activate
```

### Windows:

```bash
.venv\Scripts\activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Configure pre-commit hooks (required once)

This project uses **pre-commit** to manage Git hooks, including **nbstripout**, which automatically removes notebook outputs and metadata before commits.

This keeps commits small and prevents unnecessary notebook diffs.

Install the hooks:

```bash
pre-commit install
```

To run all hooks manually against all files:

```bash
pre-commit run --all-files
```

---

## 6. Run the notebooks

Open the project folder in VS Code and navigate to the `notebooks/` directory. The notebooks are organized by stage:

| Stage | Notebook |
|---|---|
| EDA | `EDA_Feature_Distributions.ipynb`, `EDA_Missing_Values_Analysis.ipynb` |
| Preprocessing | `Encode_Categorical_Values.ipynb`, `Validate_Interim_Dataset.ipynb` |
| Spot-checking | `Spot_Checking_ModelComparison.ipynb`, `Spot_Checking_Balancing.ipynb`, `Spot_Checking_DimensionalityReduction.ipynb` |
| Optimization | `Optimize_RF_Hyperparameters.ipynb` |
| Evaluation | `Evaluate_RF_Balancing.ipynb`, `Evaluate_RF_DimensionalityReduction.ipynb` |
| Explainability | `Explainability_CKD_Selected_Model.ipynb` |

Select the `.venv` kernel and run the cells normally.

---

# Notes for Contributors

## Notebook usage

* Notebook outputs are automatically removed via the `nbstripout` pre-commit hook
* Always pull latest changes before editing notebooks

## Recommended Workflow

Typical workflow:

1. Activate virtual environment
2. Run data cleaning: `python src/perform_minimal_data_cleaning.py`
3. Run notebooks in order (EDA -> Preprocessing -> Spot-checking -> Optimization -> Evaluation -> Explainability)
4. Generate figures (saved automatically to `reports/paper/figures/`)
5. Update LaTeX sections
6. Compile the paper

---

## Dataset

The dataset used in this project is located at:

```text
data/raw/kidney_disease.csv
```

The interim dataset (after minimal cleaning) is stored at:

```text
data/interim/kidney_disease_interim.csv
```

The processed dataset (after encoding) is stored at:

```text
data/processed/kidney_disease_encoded.csv
```

Additional dataset details can be found in:

```text
data/README.md
```

---

## Source modules

| File | Purpose |
|---|---|
| `src/perform_minimal_data_cleaning.py` | Generates the interim dataset from raw data |
| `src/modeling_utils.py` | Shared utilities for model training and evaluation |
| `src/modeling_workflow.py` | Orchestrates the modeling pipeline |
| `src/visualization_utils.py` | Shared helpers for saving figures and tables from notebooks |
| `src/explainability/` | Modular explainability framework (SHAP, permutation importance, etc.) |

---

## Tests

Unit tests are located in `tests/` and can be run with:

```bash
pytest
```

---

## Paper editing

When working on the paper:

Add figures to:

```text
reports/paper/figures/
```

Edit sections in:

```text
reports/paper/sections/
```

Update references in:

```text
reports/paper/references.bib
```

---

# Paper Compilation

Instructions to compile the LaTeX paper are available at:

```text
reports/paper/README.md
```

This paper uses the **SBC conference template**.

---
