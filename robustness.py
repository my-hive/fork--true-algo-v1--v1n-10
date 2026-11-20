#!/usr/bin/env python3
"""Dev-only search diagnostics: stationary bootstrap, DSR and CSCV PBO.

Input is search_trials.jsonl, including failed and duplicate trials in trial_count.
PBO uses distinct behavior vectors. No sealed data or nomination authority.
"""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def sharpe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 30 or not np.isfinite(values).all() or values.std(ddof=1) == 0:
        raise ValueError("at least 30 finite, nonconstant daily returns required")
    return float(values.mean() / values.std(ddof=1) * np.sqrt(252))


def bootstrap(values, seed=19, samples=500):
    values = np.asarray(values, dtype=float)
    sharpe(values)
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(samples):
        chunks, remaining = [], len(values)
        while remaining > 0:
            size = min(remaining, max(1, int(rng.geometric(1 / 50))))
            start = int(rng.integers(0, len(values)))
            chunks.extend(values[(start + i) % len(values)] for i in range(size))
            remaining -= size
        sample = np.asarray(chunks)
        if sample.std(ddof=1) > 0:
            result.append(sharpe(sample))
    if not result:
        raise ValueError("bootstrap produced no variable samples")
    return {"sharpe": sharpe(values), "ci_low": float(np.percentile(result, 2.5)),
            "p_positive": float(np.mean(np.asarray(result) > 0)), "samples": len(result)}


def dsr(values, trial_count, trial_sharpes):
    values = np.asarray(values, dtype=float)
    sr = sharpe(values) / np.sqrt(252)
    n = len(values)
    skew = float(np.mean((values - values.mean()) ** 3) / values.std() ** 3)
    kurt = float(np.mean((values - values.mean()) ** 4) / values.std() ** 4)
    dispersion = float(np.std(trial_sharpes, ddof=1)) / np.sqrt(252) if len(trial_sharpes) > 1 else 0.0
    if trial_count < 1:
        raise ValueError("trial_count required")
    if trial_count == 1 or dispersion == 0:
        benchmark = 0.0
    else:
        gamma = 0.5772156649015329
        normal = NormalDist()
        benchmark = dispersion * ((1-gamma)*normal.inv_cdf(1-1/trial_count)
                    + gamma*normal.inv_cdf(1-1/(trial_count*math.e)))
    variance = max(1e-12, 1 - skew*sr + (kurt-1)*sr*sr/4)
    probability = NormalDist().cdf((sr-benchmark)*math.sqrt(n-1)/math.sqrt(variance))
    return {"probability": probability, "trial_count": trial_count,
            "expected_max_benchmark": benchmark*math.sqrt(252)}


def pbo(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("PBO needs at least two complete behavior vectors")
    logits = []
    for train in combinations(range(16), 8):
        test = tuple(i for i in range(16) if i not in train)
        winner = int(np.argmax(matrix[list(train)].mean(axis=0)))
        test_scores = matrix[list(test)].mean(axis=0)
        order = np.argsort(test_scores, kind="stable")
        rank = int(np.where(order == winner)[0][0])
        omega = (rank + .5) / matrix.shape[1]
        logits.append(math.log(omega / (1-omega)))
    return {"pbo": float(np.mean(np.asarray(logits) <= 0)), "partitions": len(logits)}


def analyze(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    valid = [r for r in rows if r.get("returncode") == 0 and r.get("daily_pnl")]
    if not valid:
        raise ValueError("no successful trials with daily PnL")
    best = max(valid, key=lambda r: r["dev_score"])
    periods = sorted(set().union(*(r["daily_pnl"] for r in valid)))
    if len(periods) < 160:
        raise ValueError("at least 160 sessions required")
    unique = {r["behavior_sha256"]: r for r in valid}
    candidates = list(unique.values())
    matrix = np.asarray([[r["daily_pnl"].get(day, 0.0) for r in candidates] for day in periods])
    # Sixteen chronological, near-equal groups; CSCV selects combinations of groups.
    grouped = np.asarray([block.sum(axis=0) for block in np.array_split(matrix, 16)])
    best_daily = [best["daily_pnl"].get(day, 0.0) for day in periods]
    result = {"best_trial": best["trial"], "trials": len(rows), "distinct_behaviors": len(candidates),
              "bootstrap": bootstrap(best_daily)}
    sharpes = []
    for r in valid:
        try:
            sharpes.append(sharpe([r["daily_pnl"].get(day, 0.0) for day in periods]))
        except ValueError:
            pass
    result["dsr"] = dsr(best_daily, len(rows), sharpes)
    result["cscv"] = pbo(grouped) if len(candidates) >= 2 else {"pbo": None, "reason": "one distinct behavior"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=Path("search_trials.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("robustness.json"))
    args = parser.parse_args()
    result = analyze(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
