# AGENTS.md

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
    shots.csv                     # Jumpshot training data
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

To retrain the model:
```bash
python3 2k_ml.py                  # outputs new jumpshot_model.pkl
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
