"""Causal baseline for any symbol (full-size contracts, one contract). Not claimed profitable.
Slow 20-bar mean-cross momentum, entries 10:00-14:00 CT, flat otherwise.
Decide at bar i close using rows <= i; evaluator fills at i+1 open.
Baseline deliberately fails the risk gate until researchers model real entry risk.
"""
import numpy as np

FAMILY = "baseline-unvalidated"


def entry_risk(panel):
    return {symbol: np.full(len(df), 301.) for symbol, df in panel.items()}


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        close = df.close.to_numpy()
        sec = df.ct_sec.to_numpy()
        for indices in df.groupby('session', sort=False).indices.values():
            c, s = close[indices], sec[indices]
            for j in range(20, len(indices)):  # rolling mean excludes future rows
                if 36000 <= s[j] < 50400:
                    out[indices[j]] = int(np.sign(c[j] - c[j - 20:j].mean()))
        result[symbol] = out
    return result
