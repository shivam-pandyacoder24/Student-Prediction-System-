# Student Performance Analysis and Prediction System

**Minor Project | AI/ML with Python Training Program**

## Project overview

This project explores academic performance using the Portuguese-language course records in the UCI Student Performance dataset. It uses Python, NumPy, Pandas, Matplotlib, Seaborn and Scikit-learn to inspect the data, create the required exploratory charts and train a Linear Regression model for the final-period grade (`G3`).

The model uses four predictors: `studytime` (weekly study-time category), `absences` (number of absences), `G1` (first-period grade), and `G2` (second-period grade). `G3` is the target. Grades are on a 0-20 scale.

## Dataset and source

- Original dataset: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)
- Original authors: Paulo Cortez and Alice Silva (2008)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Included working file: `data/student_performance_dataset.csv` (649 records; selected columns only)
- Source and variable notes: [`data/source_info.md`](data/source_info.md)

The working CSV is a five-column extract of the UCI Portuguese-course file. The complete original dataset contains additional demographic and school-related attributes that are not used in this model. The dataset describes secondary-school students from two Portuguese schools; it should not be treated as a representative sample of all students.

**Important measurement notes:** `studytime` is an ordinal weekly band, not an exact number of hours. `absences` is a count, not an attendance percentage. The project does not invent conversions for either field.

## Objectives

1. Load and inspect the student performance data.
2. Check missing values and duplicate records and document cleaning decisions.
3. Create descriptive statistics and engineer study-time and absence groups for analysis.
4. Visualize performance using a bar chart, histogram, scatter plots, box plot and correlation heatmap.
5. Train a Linear Regression model to estimate `G3` from the selected historical features.
6. Evaluate performance with MAE, RMSE, R-squared and 5-fold cross-validation.
7. Save actual-versus-predicted values, charts, metrics and a reusable fitted pipeline.

## Repository structure

```text
Student-Prediction-System-/
├── data/
│   ├── student_performance_dataset.csv
│   └── source_info.md
├── notebooks/
│   └── student_performance_analysis.ipynb
├── src/
│   └── model.py
├── outputs/
│   ├── evaluation_results/
│   ├── visualizations/
│   └── linear_regression_pipeline.joblib
├── screenshots/
│   └── output_evidence_montage.svg
├── Project_Report.pdf
├── Project_Report.md
├── scripts/
│   └── build_report_pdf.py
├── .github/workflows/
│   └── build_report_pdf.yml
├── README.md
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the analysis

From the repository root:

```bash
python src/model.py
```

The script regenerates the data-cleaning summary, descriptive statistics, group summaries, model metrics, test predictions, cross-validation results, fitted model and visualizations under `outputs/`. Required charts are saved as PNG and SVG files.

To explore the project cell by cell, open `notebooks/student_performance_analysis.ipynb` in Jupyter Notebook, JupyterLab or Google Colab. Run the notebook from top to bottom.

## Model methodology

- Target: `G3` (final-period grade)
- Predictors: `studytime`, `absences`, `G1`, `G2`
- Split: 80% training / 20% held-out test data, with `random_state=42`
- Pipeline: median imputation, standard scaling and Linear Regression
- Model selection: `GridSearchCV` tests `fit_intercept=True` and `False` with 5-fold shuffled cross-validation using training data only
- Test metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared
- Cross-validation: 5-fold MAE and R-squared on the training partition

The imputer and scaler are part of a Scikit-learn pipeline. They are fitted only when the training folds are fitted, which avoids using held-out test data to estimate preprocessing parameters.

## Results from the recorded run

| Metric | Result |
|---|---:|
| Training records | 519 |
| Test records | 130 |
| Test MAE | 0.748 |
| Test RMSE | 1.171 |
| Test R-squared | 0.859 |
| 5-fold CV MAE (mean) | 0.816 |
| 5-fold CV MAE (standard deviation) | 0.107 |
| 5-fold CV R-squared (mean) | 0.841 |

The held-out results indicate that, on this random split, predictions were typically within about 0.75 grade points of the actual `G3` value. R-squared is approximately 0.859 on the test partition. These are results for this dataset and split, not a guarantee of future performance.

## Main observations

- `G2` has the strongest linear association with `G3` in the selected fields (correlation about 0.919), followed by `G1` (about 0.826).
- Weekly study-time category has a weaker positive association with `G3` (about 0.250).
- Absences have a weak negative association with `G3` (about -0.092) in this dataset.
- The highest mean final grade appears in the 5-10 hours/week study-time band; group sizes differ, and this observational pattern does not establish cause and effect.

## Cleaning decisions

The full 33-column source CSV was checked before producing this five-column extract: no complete duplicate records were found. The selected five-column extract contains repeated combinations because multiple students can share the same values for these five measures. Those repeated profiles are retained rather than mistakenly deleting distinct students. There are no missing values in the five required fields in the included dataset. Median imputation is nevertheless included in the training pipeline so that missing feature values in a future compatible input can be handled without leakage.

## Limitations and future improvements

- The dataset covers two Portuguese schools and one course; findings may not generalize to other education systems.
- Study time is recorded in broad weekly categories; absences do not provide an attendance percentage.
- G1 and G2 are strong predictors because they are earlier assessment grades. They are suitable only when those grades are already available; the model is not an early-semester predictor before those assessments.
- Observational correlations do not show that study time or absences cause a grade change.
- Future work could evaluate more schools, other courses, additional context variables, repeated random splits and models with uncertainty intervals.

## Responsible use

This model is an educational analysis aid, not a final judgment about a student's ability or potential. Predictions can be wrong and should be interpreted by educators alongside individual circumstances. Do not use one model score as the sole basis for academic decisions.

## References

Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), 5-12. Dataset: [UCI Machine Learning Repository - Student Performance](https://doi.org/10.24432/C5TG7T). Licensed under CC BY 4.0.


## Project report

The full report is available as [Project_Report.pdf](Project_Report.pdf), with its editable Markdown source in [Project_Report.md](Project_Report.md). The PDF is regenerated from the Markdown report and the repository's SVG charts by the GitHub Actions workflow when the report source changes.

## Output evidence

The charts are in `outputs/visualizations/`. A compact overview is in [screenshots/output_evidence_montage.svg](screenshots/output_evidence_montage.svg). The report's model metrics and prediction examples are also available under `outputs/evaluation_results/`.

## Demonstration

This is a notebook-based minor project rather than a deployed interactive application, so no separate live demo link is required. The executed analysis can be reviewed in the Jupyter notebook.
