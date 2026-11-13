"""Student Performance Analysis and Prediction System.
The selected UCI Portuguese-course columns use weekly study-time bands,
absence counts and two earlier-period grades to estimate the final grade.
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance_dataset.csv"
OUT = ROOT / "outputs"
PLOTS = OUT / "visualizations"
RESULTS = OUT / "evaluation_results"
FEATURES = ["studytime", "absences", "G1", "G2"]
TARGET = "G3"
STUDY_LABELS = {
    1: "<2 hours/week",
    2: "2-5 hours/week",
    3: "5-10 hours/week",
    4: ">10 hours/week",
}


def prepare_data():
    """Load, validate, clean and engineer descriptive groups."""
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. See data/source_info.md.")
    raw = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    if any(col not in raw.columns for col in required):
        raise ValueError("The CSV is missing one or more required columns.")
    missing_before = int(raw[required].isna().sum().sum())
    repeated_profiles = int(raw[required].duplicated().sum())
    df = raw[required].apply(pd.to_numeric, errors="coerce")
    rows_before = len(df)
    invalid_target = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    # Repeated combinations of the selected five columns are retained: they
    # may refer to different students. The full source has no exact duplicates.
    df["studytime_band"] = df["studytime"].map(STUDY_LABELS)
    df["absence_group"] = pd.cut(
        df["absences"],
        bins=[-1, 3, 10, np.inf],
        labels=["0-3 absences", "4-10 absences", "More than 10"],
    )
    summary = {
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": len(df),
        "missing_values_in_required_columns_before_cleaning": missing_before,
        "rows_removed_for_missing_target": invalid_target,
        "exact_duplicate_rows_in_full_source": 0,
        "repeated_profiles_in_selected_columns_retained": repeated_profiles,
        "features": FEATURES,
        "target": TARGET,
    }
    return df, summary


def make_eda(df):
    """Create descriptive statistics and the required visualizations."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    numeric = df[FEATURES + [TARGET]]
    stats = numeric.describe().T
    stats["median"] = numeric.median()
    stats.to_csv(RESULTS / "descriptive_statistics.csv")

    study_summary = (
        df.groupby("studytime_band", observed=False)[TARGET]
        .agg(["count", "mean", "median", "std"]).reset_index()
    )
    absence_summary = (
        df.groupby("absence_group", observed=False)[TARGET]
        .agg(["count", "mean", "median", "std"]).reset_index()
    )
    study_summary.to_csv(RESULTS / "studytime_group_summary.csv", index=False)
    absence_summary.to_csv(RESULTS / "absence_group_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(study_summary["studytime_band"].astype(str), study_summary["mean"])
    ax.set(title="Average final grade by weekly study-time band",
           xlabel="Weekly study-time band", ylabel="Mean G3 (0-20)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOTS / "bar_mean_grade_by_studytime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=15)
    ax.set(title="Distribution of final grades (G3)",
           xlabel="Final grade (0-20)", ylabel="Number of students")
    fig.tight_layout()
    fig.savefig(PLOTS / "histogram_final_grade.png", dpi=160)
    plt.close(fig)

    scatter_items = [
        ("studytime", "Weekly study-time band (1-4)", "scatter_studytime_vs_final_grade.png"),
        ("absences", "Number of school absences", "scatter_absences_vs_final_grade.png"),
        ("G2", "Second-period grade (G2)", "scatter_g2_vs_final_grade.png"),
    ]
    for feature, label, filename in scatter_items:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[feature], df[TARGET], alpha=0.55, s=28)
        if df[feature].nunique() > 1:
            slope, intercept = np.polyfit(df[feature], df[TARGET], 1)
            x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
            ax.plot(x_line, slope * x_line + intercept)
        ax.set(title=label + " and final grade", xlabel=label, ylabel="G3 (0-20)")
        fig.tight_layout()
        fig.savefig(PLOTS / filename, dpi=160)
        plt.close(fig)

    box_data = [
        df.loc[df["studytime_band"] == label, TARGET].dropna().to_numpy()
        for label in STUDY_LABELS.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(box_data, tick_labels=list(STUDY_LABELS.values()))
    ax.set(title="Final-grade spread by weekly study-time band",
           xlabel="Weekly study-time band", ylabel="G3 (0-20)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOTS / "boxplot_grade_by_studytime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap of selected academic variables")
    fig.tight_layout()
    fig.savefig(PLOTS / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    correlations = numeric.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    correlations.rename("correlation_with_G3").to_csv(
        RESULTS / "feature_target_correlations.csv"
    )
    return {
        "mean_G3": float(df[TARGET].mean()),
        "median_G3": float(df[TARGET].median()),
        "std_G3": float(df[TARGET].std()),
        "correlations_with_G3": {k: float(v) for k, v in correlations.items()},
    }


def train_evaluate(df):
    """Train Linear Regression with a preprocessing pipeline and no test leakage."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regression", LinearRegression()),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline,
        {"regression__fit_intercept": [True, False]},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    prediction_table = pd.DataFrame({
        "test_row_index": y_test.index,
        "actual_G3": y_test.to_numpy(),
        "predicted_G3": predictions,
        "absolute_error": np.abs(y_test.to_numpy() - predictions),
    }).sort_values("test_row_index")
    prediction_table.to_csv(RESULTS / "test_predictions.csv", index=False)
    pd.DataFrame({
        "fold": np.arange(1, 6), "MAE": cv_mae, "R2": cv_r2
    }).to_csv(RESULTS / "cross_validation_results.csv", index=False)
    pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].to_csv(RESULTS / "hyperparameter_search_results.csv", index=False)

    metrics = {
        "model": "Linear Regression",
        "input_features": FEATURES,
        "target": TARGET,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": 42,
        "test_MAE": mae,
        "test_RMSE": rmse,
        "test_R2": r2,
        "cross_validation_MAE_mean": float(cv_mae.mean()),
        "cross_validation_MAE_std": float(cv_mae.std(ddof=1)),
        "cross_validation_R2_mean": float(cv_r2.mean()),
        "best_parameters": search.best_params_,
        "best_cv_MAE": float(-search.best_score_),
    }
    (RESULTS / "evaluation_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    joblib.dump(model, OUT / "linear_regression_pipeline.joblib")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.7)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], "r--", label="Ideal prediction")
    ax.set(title="Actual versus predicted final grades",
           xlabel="Actual G3", ylabel="Predicted G3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)
    return metrics


def main():
    df, cleaning = prepare_data()
    (RESULTS / "cleaning_summary.json").write_text(
        json.dumps(cleaning, indent=2), encoding="utf-8"
    )
    eda = make_eda(df)
    metrics = train_evaluate(df)
    (RESULTS / "project_results.json").write_text(
        json.dumps({"cleaning": cleaning, "eda": eda, "model_evaluation": metrics}, indent=2),
        encoding="utf-8",
    )
    print("Rows:", len(df))
    print("Mean / median / SD of G3:",
          round(eda["mean_G3"], 3), round(eda["median_G3"], 3), round(eda["std_G3"], 3))
    print("MAE:", round(metrics["test_MAE"], 3))
    print("RMSE:", round(metrics["test_RMSE"], 3))
    print("R2:", round(metrics["test_R2"], 3))
    print("5-fold CV MAE:", round(metrics["cross_validation_MAE_mean"], 3))


if __name__ == "__main__":
    main()
