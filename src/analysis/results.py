from scipy.stats import skew
import pandas as pd
import numpy as np
import json
import os

import src.logging.log as log
import src.analysis.statistics as st
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TIME_RESULTS = os.path.join(RESULTS_DIR, "times.csv")
PERFORMANCE_RESULTS = os.path.join(RESULTS_DIR, "performance.csv")
STATS_METHOD_RESULTS = os.path.join(RESULTS_DIR, "method_comparison.csv")


def get_training_time():
    print("Obtaining training times...")
    loss = pd.read_csv(log.LOSS_LOG)
    details = pd.read_csv(log.DETAILS_LOG)

    # search criteria
    reps = details["experiment"].nunique()
    model_types = details.loc[details["type"].ne("universal"), "type"].unique().tolist()
    horizons = details["horizon"].unique()

    # merged dataset
    merged = loss.merge(details, on="model", suffixes=("_loss", "_detail")).copy()
    merged["start_epoch_time"] = pd.to_datetime(
        merged["start_epoch_time"], errors="coerce"
    )
    merged["end_epoch_time"] = pd.to_datetime(merged["end_epoch_time"], errors="coerce")
    merged["duration"] = (
        merged["end_epoch_time"] - merged["start_epoch_time"]
    ).dt.total_seconds()

    training_times = []
    for h in horizons:
        filt_h = merged[merged["horizon"] == h]

        # precompute universal stats for this horizon
        uni = filt_h[filt_h["type"] == "universal"]
        uni_trial = uni.groupby("model")["duration"].sum()
        uni_avg = uni_trial.mean()
        uni_total = uni["duration"].sum()

        # compute training time per model type
        for model_type in model_types:
            ft = filt_h[filt_h["type"] == model_type]
            trial_times = ft.groupby("model")["duration"].sum()
            avg_time = trial_times.mean()
            total_time = ft["duration"].sum()

            if model_type == "fine_tuned":
                # add universal time to fine_tuned
                avg_time = (0 if pd.isna(avg_time) else avg_time) + (
                    0 if pd.isna(uni_avg) else uni_avg
                )
                total_time = total_time + uni_total

            training_times.append(
                {
                    "model_type": model_type,
                    "horizon": h,
                    "per_model": float(0 if pd.isna(avg_time) else avg_time) / reps,
                    "total": float(total_time) / reps,
                }
            )

    pd.DataFrame(training_times).to_csv(TIME_RESULTS, index=False)


def get_performance_metrics(metric_names):
    print("Obtaining performance metrics...")

    metrics = pd.read_csv(log.METRIC_LOG)
    metrics = metrics.drop(columns=["experiment"])
    details = pd.read_csv(log.DETAILS_LOG)
    merged = metrics.merge(details, on="model", suffixes=("_metric", "_detail"))

    # exclude only universal
    model_types = (
        details.loc[~details["type"].isin(["universal"]), "type"].unique().tolist()
    )
    horizons = details["horizon"].unique()

    def per_experiment_stats(df):
        g = df.groupby("experiment", as_index=False)

        agg_spec = {}
        for m in metric_names:
            agg_spec[f"mean_{m}"] = (m, "mean")
            agg_spec[f"std_{m}"] = (m, "std")

        per_exp = g.agg(**agg_spec)

        for q, tag in [(0.25, "q1"), (0.50, "q2"), (0.75, "q3")]:
            qdf = g[metric_names].quantile(q)
            qdf = qdf.rename(columns={m: f"{tag}_{m}" for m in metric_names})
            per_exp = per_exp.merge(qdf, on="experiment")

        return per_exp

    rows = []
    for h in horizons:
        filt = merged[
            (merged["horizon_detail"] == h) & (merged["type_metric"] == "test")
        ]

        for mtype in model_types:
            sub = filt[filt["type_detail"] == mtype]
            if sub.empty:
                continue

            per_exp = per_experiment_stats(sub)

            # median across experiments of the per-experiment summaries
            summary = {"model_type": mtype, "horizon": h}
            for col in per_exp.columns:
                if col == "experiment":
                    continue
                summary[col] = per_exp[col].median().round(3)

            # best-per-experiment logic
            for m in metric_names:
                if mtype == "fine_tuned":
                    # Median the per-experiment values directly.
                    best_per_exp = sub.groupby("experiment", as_index=False)[m].median()
                    summary[f"best_{m}"] = round(best_per_exp[m].median(), 3)
                else:
                    # Multiple candidates → take the per-experiment best, then median.
                    best_per_exp = (
                        sub.sort_values(m, kind="stable").groupby("experiment").head(1)
                    )
                    summary[f"best_{m}"] = round(best_per_exp[m].median(), 3)

            rows.append(summary)

    pd.DataFrame(rows).to_csv(PERFORMANCE_RESULTS, index=False)


def get_stats_method_wise_comparison(metric_names):
    print("Obtaining statistical information...")
    m = pd.read_csv(log.METRIC_LOG)
    m = m[m["type"] == "test"]
    d_details = pd.read_csv(log.DETAILS_LOG)
    merged = m.merge(d_details, on="model", suffixes=("_metric", "_detail"))

    # search criteria
    horizons = d_details["horizon"].unique()
    baselines = (
        d_details.loc[~d_details["type"].isin(["universal", "fine_tuned"]), "type"]
        .unique()
        .tolist()
    )

    all_rows = []

    for metric in metric_names:
        # aggregate seeds per consumer×method×horizon
        g = st.seed_agg(
            merged.rename(
                columns={
                    "dataset": "consumer_id",
                    "type_detail": "method",
                    "horizon_detail": "horizon",
                }
            ),
            metric,
        )

        results = []
        for h in horizons:
            # pairwise Wilcoxon (proposed=fine_tuned vs each baseline)
            for base in baselines:
                deltas = st.deltas_for_horizon(g, h, metric, base)
                symm = float(skew(deltas, bias=False)) if deltas.size >= 3 else np.nan
                W = st.wilcoxon_blocked(deltas)
                hl, ci = st.hl_and_ci(deltas)
                results.append(
                    {
                        "row_type": "wilcoxon",
                        "metric": metric,
                        "horizon": h,
                        "baseline": base,
                        "n_consumers": int(deltas.size),
                        "skew_delta": symm,
                        "HL_effect": hl,
                        "HL_CI_low": ci[0],
                        "HL_CI_high": ci[1],
                        "p_Wilcoxon_less": W["p_less"],
                        "p_Wilcoxon_greater": W["p_greater"],
                        "p_friedman": np.nan,
                        "avg_ranks": np.nan,
                        "CD_0.05": np.nan,
                    }
                )

            # omnibus ranks across all methods
            fr = st.friedman_block(g, h, metric)
            results.append({"horizon": h, **fr})

        df = pd.DataFrame(results)

        # Holm correction (within this metric’s Wilcoxon family)
        mask = df["row_type"].eq("wilcoxon")
        for col in ["p_Wilcoxon_less", "p_Wilcoxon_greater"]:
            p = df.loc[mask, col].fillna(1.0).to_numpy()
            df.loc[mask, col + "_adj"] = multipletests(p, method="holm")[1]

        # decision column (one-sided, lower metric = better)
        if (
            "p_Wilcoxon_less_adj" in df.columns
            and "p_Wilcoxon_greater_adj" in df.columns
        ):
            df["decision"] = np.where(
                (df["row_type"].eq("wilcoxon"))
                & (df["p_Wilcoxon_less_adj"] < 0.05)
                & (df["HL_effect"] < 0),
                "better",
                np.where(
                    (df["row_type"].eq("wilcoxon"))
                    & (df["p_Wilcoxon_greater_adj"] < 0.05)
                    & (df["HL_effect"] > 0),
                    "worse",
                    np.where(df["row_type"].eq("wilcoxon"), "ns", np.nan),
                ),
            )

        all_rows.append(df)

    # concat all metrics and write once
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(STATS_METHOD_RESULTS, index=False)
