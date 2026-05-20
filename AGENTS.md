# AGENTS.md

## Project overview

NBA jumpshot prediction web app. A Python/Flask backend (`app.py`) serves a machine learning model trained on NBA 2024–25 stats and NBA 2K26 ratings. A vanilla JS frontend sends player data to the API and displays predictions.

## Project structure

```
ML_project/
  app.py                    # Flask app entry point — all API routes live here
  requirements.txt          # Python dependencies
  jumpshot_predictor/       # ML model logic (training, inference, feature engineering)
  frontend/
    index.html              # Main UI
    app.js                  # Fetch calls to backend API, DOM updates
    styles.css              # All styling
  nba_2425_stats.csv        # Raw player stats dataset
  nba2k26_ratings.csv       # NBA 2K26 ratings dataset
  output.txt                # Model output / prediction logs
```

New route handlers go in `app.py`. New model logic goes in `jumpshot_predictor/`. Do not put business logic in `frontend/app.js` — keep it to UI and API calls only.

## Stack

- **Backend**: Python 3, Flask
- **ML**: scikit-learn (infer from context) — model lives in `jumpshot_predictor/`
- **Frontend**: Vanilla JS, HTML, CSS — no framework, no build step
- **Data**: CSV files loaded at runtime (pandas)
- **Database**: SQL (see schema below)

## API routes (known)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/jumpshot/score` | Submit player data, returns jumpshot prediction score |

## Database schema

> Fill in your table names and columns here once known. Example format:

```
players     id, name, team, season
stats       id, player_id (FK → players), points, assists, rebounds, fg_pct
ratings     id, player_id (FK → players), overall, jumpshot, speed
```

Run `\dt` in psql (or equivalent) and paste results here to complete this section.

## Setup

```bash
pip install -r requirements.txt   # install Python dependencies
python3 app.py                    # start Flask dev server
```

Frontend is static — open `frontend/index.html` directly or serve via Flask.

## Running

```bash
python3 app.py        # Flask backend, default port 5000
```

Frontend: open `frontend/index.html` in browser, or configure Flask to serve it as static.

## Rules

- Use **pydantic** for all request/response validation on API endpoints
- Write **docstrings** on every function and class — include args, return type, and a one-line description
- Write a **test** for every new function in `jumpshot_predictor/` (use pytest)
- Use **type hints** on all function signatures
- Keep ML logic out of `app.py` — routes call into `jumpshot_predictor/`, never implement model logic inline
- Data loading (CSV/DB) happens once at startup, not per request
- Return consistent JSON error shapes: `{ "error": "human-readable message" }`
- No hardcoded file paths — use `pathlib.Path` relative to project root
