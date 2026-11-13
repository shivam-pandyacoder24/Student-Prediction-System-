# Student Performance Analysis and Prediction System

**Minor Project Report**  
**Student:** Shivam A Pandya  
**Program:** B.Tech - AI Engineering, SRM Ramapuram

## Abstract
This project analyzes academic records from the Portuguese-course subset of the UCI Student Performance dataset and builds a Linear Regression model to estimate final-period grade (G3). Inputs are weekly study-time category, number of absences, first-period grade (G1) and second-period grade (G2). It includes data inspection, cleaning checks, descriptive statistics, feature engineering, the requested EDA visualizations, training/testing, cross-validation, basic parameter search and regression evaluation.

## 1. Introduction and problem statement
Academic performance is associated with learning habits, attendance patterns and earlier academic results. This project uses historical student data to inspect those patterns and estimate a final examination mark. The system is an analysis aid, not a definitive assessment of an individual student.

## 2. Objectives
- Inspect and clean the student performance data.
- Calculate mean, median and standard deviation.
- Analyze studytime, absences, G1, G2 and G3.
- Generate bar, histogram, scatter, box and correlation heatmap visualizations.
- Train Linear Regression and evaluate with MAE, RMSE, R-squared and cross-validation.
- Interpret findings, limitations and possible improvements.

## 3. Dataset and source
The data is from the UCI Machine Learning Repository's Student Performance dataset, Portuguese-course file (649 records). Original authors: Paulo Cortez and Alice Silva (2008). DOI: 10.24432/C5TG7T. License: CC BY 4.0. The working CSV is a five-column extract: studytime, absences, G1, G2 and G3. See data/source_info.md for attribution.

**Variable meanings:** studytime is a weekly time band (1: <2 hours; 2: 2-5 hours; 3: 5-10 hours; 4: >10 hours); absences is an absence count (not attendance percentage); G1 and G2 are earlier-period grades; G3 is final-period grade on a 0-20 scale.

## 4. Data preprocessing and feature engineering
The selected columns contain 649 records, no missing values, and no missing target values. The complete 33-column original source was checked and contained zero exact duplicate rows. The five-column extract has 109 repeated feature combinations, retained because different students can share the same selected values. Study-time and absence groups were derived for EDA.

## 5. Exploratory data analysis
The mean G3 is 11.906, the median is 12.000 and standard deviation is 3.231. Correlations with G3 are G2: 0.919, G1: 0.826, studytime: 0.250, and absences: -0.091. Charts are provided in outputs/visualizations:
- bar_mean_grade_by_studytime.svg
- histogram_final_grade.svg
- scatter_studytime_vs_final_grade.svg
- scatter_absences_vs_final_grade.svg
- scatter_g2_vs_final_grade.svg
- boxplot_grade_by_studytime.svg
- correlation_heatmap.svg

## 6. Machine learning methodology
Target: G3. Predictors: studytime, absences, G1 and G2. An 80/20 split with random_state=42 produced 519 training rows and 130 held-out test rows. The model uses a Scikit-learn Pipeline with median imputation, standard scaling and Linear Regression. GridSearchCV tested fit_intercept=True and False using shuffled 5-fold cross-validation on the training partition only. Transformations are fitted within the pipeline to avoid test-data leakage.

## 7. Evaluation results
| Metric | Result |
|---|---:|
| Test MAE | 0.748 |
| Test RMSE | 1.171 |
| Test R-squared | 0.859 |
| 5-fold CV MAE mean | 0.816 |
| 5-fold CV MAE standard deviation | 0.107 |
| 5-fold CV R-squared mean | 0.841 |

The held-out predictions are typically within about 0.75 grade points on this split. The complete model implementation writes individual actual and predicted G3 values to outputs/evaluation_results/test_predictions.csv after running src/model.py.

## 8. Findings and interpretation
G2 has the strongest linear relationship with G3, followed by G1. The study-time category has a smaller positive association, and absences have a weak negative association. These are observed relationships, not evidence of cause and effect. G1 and G2 are known only after earlier assessments and therefore make this a later-period prediction model.

## 9. Limitations and future improvements
- The dataset covers two Portuguese schools and one course; findings may not generalize to other education systems.
- Study time is recorded in broad weekly categories; absences do not give an attendance percentage.
- G1 and G2 are strong features but make the model unsuitable before those assessments.
- Correlations do not establish causation.
- Future work could add more schools and subjects, attendance rates where recorded, additional models, repeated splits and prediction uncertainty.

## 10. Conclusion
The project implements a complete regression workflow for the UCI Portuguese-course student data. It performs data checks, descriptive analysis, required visualizations, model training, cross-validation, tuning and evaluation. The held-out MAE is 0.748, RMSE is 1.171 and R-squared is 0.859 for this split. Outputs should support investigation and not act as the sole basis for educational decisions.

## References
Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), 5-12. UCI Student Performance dataset. https://doi.org/10.24432/C5TG7T. License: CC BY 4.0.
