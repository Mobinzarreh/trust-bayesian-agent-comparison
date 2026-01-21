#!/usr/bin/env python3
"""
Compute robust, defensible parameter recommendations from existing sensitivity results.

Method:
- For each parameter, aggregate partner CSVs in results/sensitivity_<param>/
- Compute mean metric values per parameter value
- Rank values per partner and metric (mutual_coop, total_payoff: descending; betrayal_rate: ascending)
- Aggregate ranks across 5 partners × 3 metrics using mean rank (equal weights)
- Choose the value with the best (lowest) average rank as the robust recommendation

Writes recommendations to results/recommended_params.json and prints a concise table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PARAMS = [
    "eta",
    "memory_discount",
    "trust_discount",
    "trust_smoothing",
    "loss_aversion",
    "lambda_surprise",
    "inverse_temperature",
    "initial_x",
    "t_init",
]

METRICS = [
    ("mutual_coop_rate", "desc"),
    ("total_payoff", "desc"),
    ("betrayal_rate", "asc"),
]


def load_param_partner_df(param: str, partner: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / f"sensitivity_{param}" / f"{partner}_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df


def list_partners_for_param(param: str) -> List[str]:
    param_dir = RESULTS_DIR / f"sensitivity_{param}"
    if not param_dir.exists():
        return []
    return [p.stem.replace("_results", "") for p in sorted(param_dir.glob("*_results.csv"))]


def recommend_for_param(param: str) -> dict | None:
    partners = list_partners_for_param(param)
    if not partners:
        return None

    # Collect per-partner mean metrics per value
    value_stats: Dict[float, List[float]] = {}
    per_partner_tables: Dict[str, pd.DataFrame] = {}

    for partner in partners:
        df = load_param_partner_df(param, partner)
        if df.empty:
            continue
        means = df.groupby(param)[[m for m, _ in METRICS]].mean().reset_index()
        per_partner_tables[partner] = means

    if not per_partner_tables:
        return None

    # Build rank tables and aggregate mean rank
    # Collect candidate values (union across partners)
    all_values = sorted({v for means in per_partner_tables.values() for v in means[param].unique()})
    rank_sum = {v: 0.0 for v in all_values}
    rank_count = {v: 0 for v in all_values}

    for partner, means in per_partner_tables.items():
        for metric, direction in METRICS:
            # Rank: larger better => descending (rank 1 is best)
            ascending = True if direction == "asc" else False
            tmp = means[[param, metric]].copy()
            tmp["rank"] = tmp[metric].rank(ascending=ascending, method="average")
            # If descending desired, invert ranks so best gets 1
            if not ascending:
                # Convert to descending rank where 1 is best
                max_rank = tmp["rank"].max()
                tmp["rank"] = max_rank + 1 - tmp["rank"]
            for _, row in tmp.iterrows():
                v = float(row[param])
                r = float(row["rank"])
                rank_sum[v] += r
                rank_count[v] += 1

    # Compute average rank (lower is better)
    avg_rank = {v: (rank_sum[v] / rank_count[v]) for v in all_values if rank_count[v] > 0}
    best_value = min(avg_rank, key=avg_rank.get)

    # Also record simple metric means across partners at the best value
    metric_means = {}
    for metric, _ in METRICS:
        vals = []
        for means in per_partner_tables.values():
            row = means[means[param] == best_value]
            if not row.empty:
                vals.append(float(row.iloc[0][metric]))
        if vals:
            metric_means[metric] = float(np.mean(vals))

    # Prepare top-3 ranked values for transparency
    top3 = sorted(avg_rank.items(), key=lambda x: x[1])[:3]

    return {
        "parameter": param,
        "recommended": best_value,
        "avg_rank": avg_rank[best_value],
        "top3": top3,
        "metric_means": metric_means,
        "partners_included": sorted(per_partner_tables.keys()),
    }


def main() -> None:
    recommendations: Dict[str, dict] = {}
    for param in PARAMS:
        rec = recommend_for_param(param)
        if rec is not None:
            recommendations[param] = rec

    if not recommendations:
        print("No sensitivity results found. Please run scripts/run_all_sensitivity.py first.")
        return

    out_path = RESULTS_DIR / "recommended_params.json"
    with open(out_path, "w") as f:
        json.dump(recommendations, f, indent=2)

    # Print concise table
    print("\nParameter Recommendations (robust, mean-rank across metrics & partners):")
    rows = []
    for p, rec in recommendations.items():
        rows.append({
            "parameter": p,
            "recommended": rec["recommended"],
            "avg_rank": round(rec["avg_rank"], 3),
            "mutual_coop": round(rec["metric_means"].get("mutual_coop_rate", float("nan")), 3),
            "betrayal": round(rec["metric_means"].get("betrayal_rate", float("nan")), 3),
            "payoff": round(rec["metric_means"].get("total_payoff", float("nan")), 1),
        })
    table = pd.DataFrame(rows)
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print("\n" + table.to_string(index=False))

    print(f"\n✓ Saved recommendations to: {out_path}")


if __name__ == "__main__":
    main()
