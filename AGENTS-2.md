# AGENTS-2.md

## Project overview

NBA jumpshot prediction web app. A Python/Flask backend serves a machine learning model that scores and ranks jumpshots using NBA 2024–25 stats and NBA 2K26 ratings. A vanilla JS frontend sends requests to the API and displays results. The `jumpshot_predictor/` package handles all model logic, meta detection, and patch tracking.

## Project structure

```
ML_project/
  app.py                          # Flask entry point — all API routes
  2k_ml.py                        # Model training script — run to retrain jumpshot_model.pkl
  requirements.txt                # Python dependencies
  README.md
  AGENTS.md

  jumpshot_predictor/             # Core ML package
    __init__.py
    service.py                    # Inference logic — called by app.py routes
    meta_detector.py              # Detects meta jumpshots from current patch data
    jumpshot_model.pkl            # Trained model artifact (do not edit manually)
    jumpshot_green_model.pkl      # Trained green% model artifact (do not edit manually)
    shots.csv                     # Jumpshot training data
    community_research.csv        # Low-trust public/community reference signals
    exploit_leaderboard.csv       # Ranked exploit jumpshots
    exploit_chart.png             # Visualization of exploit data
    patch_alerts.csv              # Alerts triggered by patch changes
    patch_impact.csv              # Patch impact scores per jumpshot

  frontend/
    index.html                    # Main UI
    app.js                        # API calls + DOM updates (no business logic here)
    styles.css

  nba_2425_stats.csv              # Raw NBA 2024–25 player stats
  nba2k26_ratings.csv             # NBA 2K26 player ratings
  output.txt                      # Latest model output log
  output_20251222_132546.txt      # Timestamped output snapshot
```

**Where new code goes:**
- New API routes → `app.py`
- New model/inference logic → `jumpshot_predictor/service.py` or a new module in `jumpshot_predictor/`
- Model retraining changes → `2k_ml.py`
- UI changes → `frontend/`
- Never put ML logic directly in `app.py` or `frontend/app.js`

## Stack

- **Backend**: Python 3, Flask
- **ML**: scikit-learn — model saved as `jumpshot_predictor/jumpshot_model.pkl`
- **Data**: pandas, CSV files (`nba_2425_stats.csv`, `nba2k26_ratings.csv`, `shots.csv`)
- **Frontend**: Vanilla JS, HTML, CSS — no framework, no build step
- **Validation**: Pydantic

## API routes (known)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/jumpshot/score` | Submit player/jumpshot data, returns prediction score |

## Setup

```bash
pip install -r requirements.txt   # install dependencies
python3 app.py                    # start Flask dev server (default :5000)
```

To retrain the jumpshot models:
```bash
python3 jumpshot_predictor/meta_detector.py
```

## Rules

- Use **pydantic** for all API request/response validation
- Write **docstrings** on every function and class — include args, return type, one-line summary
- Write a **pytest test** for every new function in `jumpshot_predictor/`
- Use **type hints** on all function signatures
- `app.py` routes call into `jumpshot_predictor/service.py` — no inline model logic in routes
- Load CSVs and the `.pkl` model **once at startup**, not per request
- Return consistent JSON errors: `{ "error": "human-readable message" }`
- Use `pathlib.Path` for all file paths — no hardcoded strings
- Do not commit retrained `.pkl` files without updating `output.txt` with a run summary


## New Additions

### 0. Public research signals are reference-only

Public Reddit and YouTube claims may be useful for finding combos to test, but they must not be treated as ground-truth model training rows unless they include attempts, makes, greens, patch/version, player rating, and test context.

**Required handling:**
- Store public claims in `jumpshot_predictor/community_research.csv`.
- Mark each row with `source_quality` such as `lab` or `community`.
- Do not append public claims directly into `shots.csv` unless they include real session counts.
- Use community data to suggest combos to test, not to inflate exploit scores.


### 1. Green% must be tracked, not estimated

The current formula in `add_session` estimates greens from make%:

```python
green_factor = 0.65 + (player_3pt - 85) * 0.003
greens = round(make_pct * green_factor * attempts)
```

This is a hardcoded approximation that treats all jumpshots equally. It must be replaced with real user-tracked green counts.

**Required change — `service.py` `add_session`:**
- Make `greens` a required field. Raise `ValueError("greens is required")` if missing or empty.
- Remove the estimation fallback entirely.
- `shots.csv` already has a `greens` column — no schema change needed.
```python
# Before (remove this)
greens_value = payload.get("greens")
if greens_value in ("", None):
    greens = round(make_pct * green_factor * attempts)  # estimation

# After
greens_value = payload.get("greens")
if greens_value in ("", None):
    raise ValueError("greens is required — count your greens each session")
greens = _to_int(greens_value)
```

---

### 2. Add `latency_score` to the exploit leaderboard

The existing `exploit_score` rewards shots that outperform expected make%. It does not reward shots with wide green windows (latency tolerance). Add a separate `latency_score` column.

**Required change — `meta_detector.py` `build_exploit_leaderboard`:**

```python
# Add after existing exploit_score calculation
exploits["green_pct"] = exploits["total_greens"] / exploits["total_attempts"]
exploits["latency_score"] = (
    exploits["green_pct"]
    / exploits["make_pct"].replace(0, float("nan"))  # avoid divide-by-zero
) * np.log1p(exploits["total_attempts"])
```

- `green_pct / make_pct` — the ratio of greens to makes. High ratio = wide timing window = more latency-tolerant.
- Multiplied by `log1p(attempts)` to weight by sample size, same as `exploit_score`.
- Save this column to `exploit_leaderboard.csv` so `service.py` can serve it.
**Also update `_parse_leaderboard` in `service.py`** to parse `latency_score` from the CSV and return it in `score_jumpshot` responses.

---

### 3. Train a second XGBoost target on `green_pct`

The model in `meta_detector.py` currently trains one pipeline predicting `make_pct`. Add a second pipeline trained on `green_pct`. The residual from this model is the latency edge signal.

**Required change — `meta_detector.py`:**

```python
# In run(), after training the make_pct model:
model_green, metrics_green = train_baseline(shots, target="green_pct")
joblib.dump(model_green, DATA_DIR / "jumpshot_green_model.pkl")

shots["pred_green"] = model_green.predict(shots[CAT_COLS + NUM_COLS])
shots["green_residual"] = shots["green_pct"] - shots["pred_green"]
```

Update `train_baseline` to accept a `target` parameter (default `"make_pct"`):

```python
def train_baseline(shots: pd.DataFrame, target: str = "make_pct") -> tuple[Pipeline, dict]:
    X = shots[CAT_COLS + NUM_COLS]
    y = shots[target]
    # rest unchanged
```

Pass `green_residual` into `build_exploit_leaderboard` alongside `residual` so both signals are available per combo.

**New file produced:** `jumpshot_predictor/jumpshot_green_model.pkl`
Add this to the project structure and load it at startup in `service.py` alongside the existing model.

---

### Summary of files changed

| File | Change |
|------|--------|
| `jumpshot_predictor/service.py` | Make `greens` required in `add_session`; parse + return `latency_score` and `green_residual` |
| `jumpshot_predictor/meta_detector.py` | Add `latency_score` to leaderboard; train second model on `green_pct`; add `green_residual` column |
| `jumpshot_predictor/exploit_leaderboard.csv` | New columns: `latency_score`, `green_pct` |
| `jumpshot_predictor/jumpshot_green_model.pkl` | New artifact — green% baseline model |
| `frontend/app.js` | Display `latency_score` alongside `exploit_score` in UI |
