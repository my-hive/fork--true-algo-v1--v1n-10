#!/usr/bin/env python3
"""Bounded Optuna search over @parameter@ slots in strategy.py.

Development data only. Every trial is recorded; identical trade ledgers collapse
onto one behavior. This tool proposes candidates; it never authorizes sealing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import optuna

MAX_TRIALS = 64
MAX_PARAMS = 16
NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    space = spec.get("space")
    if not isinstance(space, dict) or not 1 <= len(space) <= MAX_PARAMS:
        raise ValueError("space must contain 1..16 parameters")
    for name, rule in space.items():
        if not NAME.fullmatch(name) or not isinstance(rule, dict):
            raise ValueError(f"invalid parameter: {name}")
        kind = rule.get("type")
        if kind in ("int", "float"):
            low, high = rule.get("low"), rule.get("high")
            if type(low) not in (int, float) or type(high) not in (int, float) or not low < high:
                raise ValueError(f"invalid bounds: {name}")
            if kind == "int" and (type(low) is not int or type(high) is not int):
                raise ValueError(f"integer bounds required: {name}")
            if "step" in rule and (type(rule["step"]) not in (int, float) or rule["step"] <= 0):
                raise ValueError(f"invalid step: {name}")
        elif kind == "categorical":
            if not rule.get("choices") or len(rule["choices"]) > 32 or any(
                    not isinstance(x, (str, int, float, bool)) for x in rule["choices"]):
                raise ValueError(f"invalid choices: {name}")
        else:
            raise ValueError(f"unknown parameter type: {name}")
    trials = spec.get("trials", 16)
    if type(trials) is not int or not 1 <= trials <= MAX_TRIALS:
        raise ValueError("trials must be an integer in 1..64")
    if type(spec.get("seed", 19)) is not int:
        raise ValueError("seed must be an integer")
    if type(spec.get("timeout_seconds", 900)) is not int or not 1 <= spec.get("timeout_seconds", 900) <= 1800:
        raise ValueError("timeout_seconds must be an integer in 1..1800")
    data = spec.get("data", "/data/dev")
    if not isinstance(data, str) or not data.startswith("/"):
        raise ValueError("data must be an absolute development-data path")
    symbols = spec.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(
            not isinstance(s, str) or not s.isalnum() for s in symbols):
        raise ValueError("symbols must be nonempty root names")
    return spec


def render(template, values):
    used = set()
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f"undeclared placeholder: {name}")
        used.add(name)
        return str(values[name])
    source = re.sub(r"@([a-z][a-z0-9_]*)@", replace, template)
    if used != set(values):
        raise ValueError("every declared parameter must occur in strategy.py")
    return source


def metrics(report):
    s = report["stressed"]
    return {k: s.get(k) for k in ("dev_score", "worst_three_mean", "positive_year_fraction", "total_pnl")}


def daily_pnl(path):
    days = {}
    with path.open() as source:
        for line in source:
            fill = json.loads(line)
            day = days.setdefault(fill["session"], 0.0)
            days[fill["session"]] = day + fill["realized_pnl"] - fill["commission"]
    return days


def run(spec_path):
    spec = load_spec(spec_path)
    template = Path("strategy.py").read_text()
    records = Path("search_trials.jsonl")
    seen = set()
    # One log per campaign: a different spec archives the old log so DSR/PBO count
    # only this campaign's trials. Same spec resumes (Optuna load_if_exists).
    stamp = Path("search_trials.spec")
    spec_sha = hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
    if records.exists() and (not stamp.exists() or stamp.read_text() != spec_sha):
        records.rename(f"search_trials.{int(time.time())}.jsonl")
    stamp.write_text(spec_sha)
    if records.exists():
        for line in records.read_text().splitlines():
            row = json.loads(line)
            if row.get("behavior_sha256"):
                seen.add(row["behavior_sha256"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend("search_journal.json"))
    study = optuna.create_study(direction="maximize", storage=storage,
                                study_name=hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()[:16],
                                sampler=optuna.samplers.TPESampler(seed=spec.get("seed", 19)),
                                load_if_exists=True)

    def objective(trial):
        values = {}
        for name, rule in spec["space"].items():
            kind = rule["type"]
            if kind == "categorical":
                values[name] = trial.suggest_categorical(name, rule["choices"])
            else:
                values[name] = getattr(trial, "suggest_" + kind)(name, rule["low"], rule["high"],
                    **({"step": rule["step"]} if "step" in rule else {}))
        source = render(template, values)
        sha = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            tmp = Path(tmp)
            candidate, report, ledger = tmp/"candidate.py", tmp/"report.json", tmp/"fills.jsonl"
            candidate.write_text(source)
            command = ["bash", "eval/eval.sh", "--data", spec.get("data", "/data/dev"), "--symbols", *spec["symbols"],
                       "--strategy", str(candidate), "--output", str(report), "--ledger", str(ledger),
                       "--null-seeds", "0", "--mode", "dev"]
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=spec.get("timeout_seconds", 900),
                                  env={**os.environ, "PYTHON": sys.executable})
            row = {"trial": trial.number, "params": values, "source_sha256": sha,
                   "returncode": proc.returncode}
            if proc.returncode == 0:
                result = json.loads(report.read_text())
                behavior = hashlib.sha256(ledger.read_bytes()).hexdigest()
                row.update(metrics(result), behavior_sha256=behavior,
                           daily_pnl=daily_pnl(ledger), duplicate_behavior=behavior in seen)
                seen.add(behavior)
                value = -1.0 if row["duplicate_behavior"] else float(row["dev_score"])
            else:
                row["error"] = proc.stdout[-2000:]
                value = -2.0
            with records.open("a") as out:
                out.write(json.dumps(row, allow_nan=False) + "\n")
            return value

    study.optimize(objective, n_trials=spec["trials"])
    best = Path("search_best.py")
    best.write_text(render(template, study.best_params))
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params,
                      "best_source_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
                      "best_source": str(best), "trials_file": str(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("campaign.json"))
    args = parser.parse_args()
    run(args.spec)
