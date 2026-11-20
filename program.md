# `true-algo-hive` — Swarm Task Contract & Research Protocol

## 1. Objective
Discover, engineer, and validate a causal trading strategy built around the **True Algo** indicator features (`ta_box_*`, `ta_label_*`) that **regularly passes the TopstepX \$50K Combine** across rolling start dates under stressed execution costs.

Prior experiments relied exclusively on slow MNQ intraday momentum and failed out-of-sample validation. **You are explicitly mandated to explore faster intraday cadences, multi-asset opportunities (full-size contracts only: `NQ`, `ES`, `YM`, `RTY`, `CL`, `NG`, `HG`, `GC`, `SI`, `6B`, `6E`, `6J`), cross-market lead/lag relationships, and GPU-accelerated ML (`XGBoost` / `PyTorch` on RTX 4090).**

---

## 2. Repository & Sandbox Rules (Strict)
1. **Editable File:** You may ONLY edit `strategy.py` (and `campaign.json` when running Optuna searches). Never modify `eval/sim.py`, `walkforward.py`, `robustness.py`, or `board.py`.
2. **Data Location:** Read-only Parquet datasets live at `/data/dev/<ASSET>_1m_ta.parquet`.
3. **Sealed Validation Off-Limits:** You do not have access to `/data/validation`. Never attempt to probe, locate, or infer sealed validation paths. Only the Queen agent may submit verified finalists to the validator socket.
4. **No Stale Bytecode:** Always execute evaluations via `PYTHONDONTWRITEBYTECODE=1 bash eval/eval.sh`.

---

## 3. Strategy Interface (`strategy.py`)
Your `strategy.py` must expose:

```pytimport numpy as np

def signals(panel):
    """panel: {symbol: DataFrame}. Index `ts` (UTC, minute START), columns open/high/low/close/volume,
    `session` (str CME session label, rolls 17:00 CT), `ct_sec` (seconds since midnight CT),
    `contract`, plus every True Algo / cross-asset column listed in section 4.
    Return {symbol: int64 array, len(df)} of TARGET contracts (positive long, negative short).
    Full-size contracts: |target| * standard-equivalent weight must stay <= 5 per symbol (1 per NQ/ES/...)."""
    return {s: np.zeros(len(df), dtype=np.int64) for s, df in panel.items()}

def entry_risk(panel):
    """{symbol: float array, len(df)} dollar risk of a position opened at that bar (stop distance x point value x size).
    Must be honest; eligibility requires entry risk <= $300 and the evaluator checks it against real bar ranges."""
    return {s: np.full(len(df), 301.) for s, df in panel.items()}
```

Run exactly: `PYTHONDONTWRITEBYTECODE=1 bash eval/eval.sh --data /data/dev --symbols NQ ES --strategy strategy.py --output report.json --ledger fills.jsonl --null-seeds 0 --mode dev`
(`--symbols` takes real symbols present in `/data/dev`; one eval on a deep-history symbol takes ~2 minutes, so batch your thinking, not your trials.)
`dev_score` in the printed JSON is the graded leaderboard number; `score`/`eligible` are the strict gate.

### Causality Contract (Zero Tolerance)
- **Bar `i` signal uses ONLY rows `<= i`.**
- Signals computed at the close of 1m bar `i` are filled by the simulator at **bar `i + 1` open** plus slippage.
- **Forbidden:** `shift(-1)`, centered rolling windows (`center=True`), full-series `.mean()` / `.quantile()` / `StandardScaler().fit(df)`, or any future bar indexing.
- **Automated Prefix Trap:** `eval/sim.py` truncates the panel at fixed UTC cutoffs plus sampled position-change bars and asserts targets on the prefix equal targets on the full data. Any mismatch aborts the run.

---

## 4. Available Dataset Columns (`/data/dev/<ASSET>_1m_ta.parquet`)
All `ta_*` columns are derived from closed 15m/30m True Algo bars and forward-filled causally onto the 1m grid:

- **Execution & Session:** `ts_utc`, `ts_ct`, `session_date`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- **True Algo Box Geometry:**
  - `ta_box_upper`, `ta_box_lower`, `ta_box_mid`, `ta_box_width`
  - `ta_box_dir` (`1` bullish box, `-1` bearish box, `0` neutral)
  - `ta_box_age` (number of higher-timeframe bars since current box formed)
  - `ta_active_zone_count` (count of active True Algo zones)
- **Normalized Box Features (Scale-Invariant):**
  - `ta_close_vs_mid_pct`: `(close - ta_box_mid) / ta_box_mid`
  - `ta_box_pos`: `(close - ta_box_lower) / ta_box_width` (`0.0` = at lower boundary, `1.0` = at upper boundary, `<0` or `>1` = outside box)
  - `ta_box_width_pct`: `ta_box_width / ta_box_mid`
- **True Algo Labels:**
  - `ta_label_code` (`1` buy/bull label, `-1` sell/bear label, `2` flip/reversal, `0` none)
  - `ta_bars_since_label` (15m/30m bars elapsed since last non-zero label)
- **Cross-Asset Causal Context (1m aligned):**
  - `{nq,es,cl,gc}_box_dir`, `{nq,es,cl,gc}_box_pos`, `{nq,es,cl,gc}_ret_15m` (a symbol's own cross-asset columns are omitted). Load another symbol's own file via `signals(panel)` by requesting it in `--symbols` and using `panel['ES']`; there is no DX data.

---

## 5. TopstepX $50K Combine Rules (Enforced in `eval/sim.py`)
- **Starting Balance:** `$50,000`
- **Profit Target:** `$3,000` (`$53,000` equity)
- **Daily Loss Limit (DLL):** `$1,000` intrabar (breach flattens immediately and locks out the rest of the session)
- **Trailing Max Drawdown:** `$2,000` End-of-Day (EOD) high-water mark (locks at `$50,000` once high-water mark reaches `$52,000`)
- **Max Position Size:** `MAX_EXPOSURE = 5` standard-equivalent contracts per symbol (full-size: NQ `$5.00/tick`, ES `$12.50/tick`, CL `$10/tick`, GC `$10/tick`; see `eval/instruments.py`)
- **Max Entry Risk:** `$300` per trade stop threshold
- **Minimum Trading Days:** `>= 2` days
- **Consistency Rule:** Best single session PnL must be `< 50%` of total campaign profit
- **Session Flatten:** All positions force-closed at `15:09 CT` (no overnight holds)
- **Stressed Costs:** 2 ticks slippage per side and 1.5x the estimated per-side fee from `eval/instruments.py` (the evaluator reports base and stressed; stressed is authoritative)

---

## 6. Five Mandatory Strategy Families
Every worker is seeded to a distinct hypothesis family. **Do not repeat falsified slow MNQ momentum.** Tag every proposal on `board.py` with its `FAMILY` ID:

1. **`FAMILY_A` — Intraday Box Boundary Rejection & Breakout (High Cadence):**
   - Trade 1m/5m price rejections at `ta_box_pos ≈ 0.0` / `1.0` or volume-confirmed breakouts outside `ta_box_upper` / `ta_box_lower` during RTH (`08:30–15:00 CT`). Target 3–12 trades/day with tight stops.
2. **`FAMILY_B` — Box Compression & Volatility Expansion:**
   - Detect regimes where `ta_box_width_pct` compresses into its bottom rolling percentile while `ta_box_age` matures, entering on range expansion across `NQ`, `ES`, `CL`, or `GC`.
3. **`FAMILY_C` — Cross-Asset True Algo Lead/Lag:**
   - Use True Algo box flips (`nq_box_dir`, `dx_box_dir`, `cl_box_dir`) or cross-market box divergences (`nq_box_pos - es_box_pos`) to trigger faster entries on lagging indices or metals.
4. **`FAMILY_D` — Adaptive Box-Anchored Risk & Scaling:**
   - Use True Algo box edges (`ta_box_lower`, `ta_box_upper`, `ta_box_mid`) as dynamic structural stop-loss and partial take-profit levels, scaling contract size (`1..5` standard-equivalents) dynamically based on remaining TopstepX drawdown headroom.
5. **`FAMILY_E` — GPU ML / Meta-Labeling (`XGBoost` / `PyTorch` on RTX 4090):**
   - Use gradient boosting or neural meta-labeling to filter True Algo box states and predict positive expectancy within 15–60 minutes.
   - **Strict ML Causality Rule:** Model training, feature scaling, and threshold selection MUST run **strictly inside past-only walk-forward windows** (`walkforward.py`). Never fit a model across the full `/data/dev` dataset.

---

## 7. Optuna Search & Anti-Overfit Acceptance Gates
1. **Bounded Campaigns:** Max **12 tunable parameters** (`@param@` placeholders) and max **64 Optuna trials** per `campaign.json`. Changing `campaign.json` automatically archives previous trial logs so Deflated Sharpe Ratio (`DSR`) trial counts remain honest.
2. **Complexity Cap:** Max 6 active logical gates/conditions per strategy. Hardcoded calendar dates, specific day-of-week bans, or holiday filters are strictly forbidden.
3. **Mandatory Dev Robustness Chain (Required before `board.py propose`):**
   - **Walk-Forward (`walkforward.py`):** 126-session rolling out-of-sample folds; `>= 60%` of folds profitable AND aggregate OOS stressed PnL `> 0`.
   - **Statistical Significance (`robustness.py`):** `DSR >= 0.95`, Stationary Bootstrap `CI_low > 0`, and 16-block CSCV `PBO <= 0.50`.
   - **Null & Perturbation Checks:** 200-seed random-entry / date-scramble null `p <= 0.05`, plus `±10%` parameter perturbation without PnL cliffs.
   - **Cost Stress:** Positive stressed PnL at `2x` slippage and `3x` commission.


## 9. True Algo session-level invariant and resampling mandate
`ta_box_*` and `ta_label_*` are session-level invariant levels, valid across every chart timeframe. Agents may resample causal 1m OHLCV inside `signals(panel)` to `1m`, `3m`, `5m`, `15m`, `30m`, or `1h`; combine True Algo levels with indicators, cross-asset features, or GPU `XGBoost`/`PyTorch` models on RTX 4090. Every calculation must use rows `<= i`, then map targets back onto the 1m grid.
