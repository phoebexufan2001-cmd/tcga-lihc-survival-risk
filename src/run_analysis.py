"""Build and internally validate a TCGA-LIHC gene-expression survival model."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RANDOM_STATE = 42


def load_cohort():
    expression = pd.read_csv(
        RAW / "TCGA.LIHC.HiSeqV2.gz", sep="\t", index_col=0
    )
    clinical = pd.read_csv(RAW / "LIHC_clinicalMatrix.tsv", sep="\t")

    tumor_samples = [
        sample for sample in expression.columns
        if len(sample) >= 15 and sample[13:15] == "01"
    ]
    expression = expression[tumor_samples]
    expression.columns = [sample[:12] for sample in expression.columns]
    expression = expression.T.groupby(level=0).first()

    clinical = clinical.drop_duplicates("_PATIENT").set_index("_PATIENT")
    clinical["duration"] = pd.to_numeric(
        clinical["days_to_death"], errors="coerce"
    ).fillna(pd.to_numeric(clinical["days_to_last_followup"], errors="coerce"))
    clinical["event"] = (
        clinical["vital_status"].astype(str).str.upper() == "DECEASED"
    ).astype(int)
    clinical = clinical.loc[clinical["duration"].notna() & (clinical["duration"] > 30)]

    shared = expression.index.intersection(clinical.index)
    expression = expression.loc[shared].apply(pd.to_numeric, errors="coerce")
    clinical = clinical.loc[shared, ["duration", "event", "pathologic_stage"]]
    expression = expression.loc[:, expression.notna().all()]

    return expression, clinical


def screen_genes(train_expression, train_survival, top_variance=500, keep=30):
    variances = train_expression.var().sort_values(ascending=False)
    candidates = variances.head(top_variance).index
    records = []

    for gene in candidates:
        values = train_expression[gene]
        median = values.median()
        high = values >= median
        if high.nunique() < 2:
            continue
        result = logrank_test(
            train_survival.loc[high, "duration"],
            train_survival.loc[~high, "duration"],
            train_survival.loc[high, "event"],
            train_survival.loc[~high, "event"],
        )
        records.append({"gene": gene, "logrank_p": result.p_value})

    return (
        pd.DataFrame(records)
        .sort_values("logrank_p")
        .head(keep)
        .reset_index(drop=True)
    )


def fit_penalized_cox(train_x, train_survival):
    penalties = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for penalty in penalties:
        fold_scores = []
        for fit_index, valid_index in folds.split(train_x, train_survival["event"]):
            fit_x = train_x.iloc[fit_index]
            valid_x = train_x.iloc[valid_index]
            fit_survival = train_survival.iloc[fit_index]
            valid_survival = train_survival.iloc[valid_index]
            fit_data = fit_x.join(fit_survival[["duration", "event"]])

            try:
                model = CoxPHFitter(penalizer=penalty, l1_ratio=0.9)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(fit_data, duration_col="duration", event_col="event")
                risk = model.predict_partial_hazard(valid_x)
                score = concordance_index(
                    valid_survival["duration"],
                    -risk,
                    valid_survival["event"],
                )
                fold_scores.append(score)
            except Exception:
                fold_scores.append(np.nan)

        full_model = CoxPHFitter(penalizer=penalty, l1_ratio=0.9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model.fit(
                train_x.join(train_survival[["duration", "event"]]),
                duration_col="duration",
                event_col="event",
            )
        nonzero_count = int((full_model.params_.abs() > 1e-4).sum())
        records.append(
            {
                "penalizer": penalty,
                "mean_cv_cindex": np.nanmean(fold_scores),
                "sd_cv_cindex": np.nanstd(fold_scores),
                "nonzero_genes": nonzero_count,
            }
        )

    tuning = pd.DataFrame(records).sort_values("mean_cv_cindex", ascending=False)
    interpretable = tuning.loc[
        (tuning["nonzero_genes"] >= 3) & (tuning["nonzero_genes"] <= 15)
    ]
    selected = interpretable.iloc[0] if len(interpretable) else tuning.iloc[0]
    best_penalty = float(selected["penalizer"])
    final_model = CoxPHFitter(penalizer=best_penalty, l1_ratio=0.9)
    final_model.fit(
        train_x.join(train_survival[["duration", "event"]]),
        duration_col="duration",
        event_col="event",
    )
    return final_model, tuning


def plot_kaplan_meier(test_survival, test_risk, threshold):
    high = test_risk >= threshold
    result = logrank_test(
        test_survival.loc[high, "duration"],
        test_survival.loc[~high, "duration"],
        test_survival.loc[high, "event"],
        test_survival.loc[~high, "event"],
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    for label, mask, color in [
        ("High risk", high, "#d62728"),
        ("Low risk", ~high, "#1f77b4"),
    ]:
        fitter = KaplanMeierFitter(label=label)
        fitter.fit(
            test_survival.loc[mask, "duration"] / 365.25,
            test_survival.loc[mask, "event"],
        )
        fitter.plot_survival_function(ax=axis, ci_show=True, color=color)
    axis.set(
        title=f"Test-set overall survival (log-rank p={result.p_value:.3g})",
        xlabel="Years",
        ylabel="Survival probability",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "test_set_kaplan_meier.png", dpi=220)
    plt.close(figure)
    return float(result.p_value), int(high.sum()), int((~high).sum())


def plot_coefficients(model):
    coefficients = model.params_.sort_values()
    nonzero = coefficients[coefficients.abs() > 1e-4]
    shown = nonzero if len(nonzero) <= 15 else nonzero.iloc[np.argsort(nonzero.abs())[-15:]]
    colors = ["#1f77b4" if value < 0 else "#d62728" for value in shown]
    figure, axis = plt.subplots(figsize=(7, max(4, len(shown) * 0.35)))
    axis.barh(shown.index, shown.values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Penalized Cox model coefficients", xlabel="Coefficient")
    figure.tight_layout()
    figure.savefig(FIGURES / "model_coefficients.png", dpi=220)
    plt.close(figure)
    return nonzero


def plot_risk_distribution(test_survival, test_risk, threshold):
    order = np.argsort(test_risk.to_numpy())
    sorted_risk = test_risk.iloc[order].reset_index(drop=True)
    sorted_events = test_survival.iloc[order]["event"].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(
        np.arange(len(sorted_risk)),
        sorted_risk,
        c=sorted_events.map({0: "#1f77b4", 1: "#d62728"}),
        s=24,
        alpha=0.8,
    )
    axis.axhline(threshold, color="black", linestyle="--", label="Training median")
    axis.set(
        title="Test-set predicted risk distribution",
        xlabel="Patients ordered by predicted risk",
        ylabel="Partial hazard",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "test_set_risk_distribution.png", dpi=220)
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    expression, survival = load_cohort()
    train_ids, test_ids = train_test_split(
        expression.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=survival["event"],
    )
    train_expression = expression.loc[train_ids]
    test_expression = expression.loc[test_ids]
    train_survival = survival.loc[train_ids]
    test_survival = survival.loc[test_ids]

    screening = screen_genes(train_expression, train_survival)
    genes = screening["gene"].tolist()
    scaler = StandardScaler()
    train_x = pd.DataFrame(
        scaler.fit_transform(train_expression[genes]),
        index=train_ids,
        columns=genes,
    )
    test_x = pd.DataFrame(
        scaler.transform(test_expression[genes]),
        index=test_ids,
        columns=genes,
    )

    model, tuning = fit_penalized_cox(train_x, train_survival)
    train_risk = model.predict_partial_hazard(train_x)
    test_risk = model.predict_partial_hazard(test_x)
    train_cindex = concordance_index(
        train_survival["duration"], -train_risk, train_survival["event"]
    )
    test_cindex = concordance_index(
        test_survival["duration"], -test_risk, test_survival["event"]
    )
    threshold = float(train_risk.median())

    logrank_p, high_count, low_count = plot_kaplan_meier(
        test_survival, test_risk, threshold
    )
    nonzero = plot_coefficients(model)
    plot_risk_distribution(test_survival, test_risk, threshold)

    screening.to_csv(TABLES / "training_gene_screen.csv", index=False)
    tuning.to_csv(TABLES / "penalty_tuning.csv", index=False)
    pd.DataFrame(
        {"gene": nonzero.index, "coefficient": nonzero.values}
    ).sort_values("coefficient", key=abs, ascending=False).to_csv(
        TABLES / "model_coefficients.csv", index=False
    )

    summary = {
        "cohort_patients": int(len(expression)),
        "cohort_events": int(survival["event"].sum()),
        "train_patients": int(len(train_ids)),
        "test_patients": int(len(test_ids)),
        "selected_genes": int(len(nonzero)),
        "best_penalizer": float(model.penalizer),
        "train_cindex": float(train_cindex),
        "test_cindex": float(test_cindex),
        "test_logrank_p": logrank_p,
        "test_high_risk_patients": high_count,
        "test_low_risk_patients": low_count,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
