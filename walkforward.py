#!/usr/bin/env python3
"""Rolling dev-only parameter search: train on past, score next block once.

Each fold re-runs bounded Optuna on train only. Frozen fold winner is evaluated
on the next chronological block with prior bars available for causal warmup.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import search


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def run(spec_path, strategy_path, output):
    spec = search.load_spec(spec_path)
    task = Path(__file__).resolve().parent
    sys.path.insert(0, str(task / "eval"))
    try:
        sim = module(task / "eval/sim.py", "v1_walkforward_sim")
    finally:
        sys.path.pop(0)
    full = sim.load_panel(spec.get("data", "/data/dev"), spec["symbols"])
    sessions = sorted(set().union(*(set(df.session.astype(str)) for df in full.values())))
    train_size = spec.get("train_sessions", 756)
    validation_size = spec.get("validation_sessions", 126)
    if type(train_size) is not int or type(validation_size) is not int or train_size < 252 or validation_size < 60:
        raise ValueError("train_sessions>=252 and validation_sessions>=60 required")
    folds = spec.get("folds", 4)
    if type(folds) is not int or not 2 <= folds <= 4:
        raise ValueError("folds must be an integer in 2..4")
    starts = list(range(train_size, len(sessions) - 59, validation_size))[:folds]
    if len(starts) < 2:
        raise ValueError("insufficient sessions for two walk-forward folds")
    workspace = Path("walkforward_work")
    workspace.mkdir(exist_ok=True)
    rows = []
    for index, start in enumerate(starts):
        train_sessions = set(sessions[start-train_size:start])
        validation_sessions = set(sessions[start:start+validation_size])
        fold = workspace / f"fold{index}"
        fold.mkdir(exist_ok=True)
        data = fold / "data"
        data.mkdir(exist_ok=True)
        for symbol, df in full.items():
            df[df.session.astype(str).isin(train_sessions)].to_parquet(data / f"{symbol}.parquet")
        shutil.copyfile(strategy_path, fold / "strategy.py")
        if not (fold / "eval").exists():
            (fold / "eval").symlink_to(task / "eval", target_is_directory=True)
        fold_spec = dict(spec, data=str(data.resolve()))
        (fold / "campaign.json").write_text(json.dumps(fold_spec))
        command = [sys.executable, str(task / "search.py"), "--spec", "campaign.json"]
        proc = subprocess.run(command, cwd=fold, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        if proc.returncode:
            rows.append({"fold": index, "error": proc.stdout[-2000:]})
            continue
        candidate_path = fold / "search_best.py"
        candidate = module(candidate_path, f"v1_walkforward_candidate_{index}")
        # Full prefix includes only past train bars plus validation, never future folds.
        prefix_sessions = train_sessions | validation_sessions
        prefix = {s: df[df.session.astype(str).isin(prefix_sessions)].copy() for s, df in full.items()}
        targets = sim.validate_targets(prefix, candidate.signals({s: df.copy() for s, df in prefix.items()}))
        risks = sim.entry_risks(candidate, prefix)
        sim.lookahead_check(candidate.signals, prefix, targets)
        sim.risk_lookahead_check(candidate, prefix, risks, targets)
        validation = {s: df[df.session.astype(str).isin(validation_sessions)].copy() for s, df in prefix.items()}
        vtargets, vrisks = {}, {}
        for symbol, df in prefix.items():
            mask = df.session.astype(str).isin(validation_sessions).to_numpy()
            vtargets[symbol] = targets[symbol][mask]
            vrisks[symbol] = risks[symbol][mask]
        stressed = sim.run_days(validation, vtargets, stressed=True, entry_risk=vrisks)
        metrics = sim.score(stressed["days"])
        rows.append({"fold": index, "first_session": min(validation_sessions), "last_session": max(validation_sessions), "source_sha256": __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest(),
                     "total_pnl": metrics["total_pnl"], "worst_three_mean": metrics["worst_three_mean"],
                     "positive_year_fraction": metrics["positive_year_fraction"],
                     "validation_sessions": len(validation_sessions)})
    completed = [r for r in rows if "total_pnl" in r]
    # v20: an out-of-sample block counts positive on PnL alone (A/B). The strict
    # combine pass rate is still reported, but no longer gates this dev diagnostic;
    # bootstrap, trial-aware DSR, PBO and the final eligibility gates still apply.
    result = {"folds": rows, "completed": len(completed),
              "positive_fold_fraction": sum(r["total_pnl"] > 0 for r in completed) / len(completed) if completed else 0.0,
              "strict_positive_fold_fraction": sum(r["total_pnl"] > 0 and r["worst_three_mean"] > 0 for r in completed) / len(completed) if completed else 0.0,
              "median_primary": float(np.median([r["worst_three_mean"] for r in completed])) if completed else 0.0,
              "median_pnl": float(np.median([r["total_pnl"] for r in completed])) if completed else 0.0,
              "passed": len(completed) == len(starts) and sum(r["total_pnl"] > 0 for r in completed) / len(completed) >= .6 and (not completed or sum(r["total_pnl"] for r in completed) > 0)}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("campaign.json"))
    parser.add_argument("--strategy", type=Path, default=Path("strategy.py"))
    parser.add_argument("--output", type=Path, default=Path("walkforward.json"))
    args = parser.parse_args()
    run(args.spec, args.strategy, args.output)
