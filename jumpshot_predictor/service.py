from __future__ import annotations

import csv
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
SHOTS_PATH = DATA_DIR / "shots.csv"
MODEL_PATH = DATA_DIR / "jumpshot_model.pkl"
GREEN_MODEL_PATH = DATA_DIR / "jumpshot_green_model.pkl"
LEADERBOARD_PATH = DATA_DIR / "exploit_leaderboard.csv"
PATCH_IMPACT_PATH = DATA_DIR / "patch_impact.csv"
PATCH_ALERTS_PATH = DATA_DIR / "patch_alerts.csv"
COMMUNITY_RESEARCH_PATH = DATA_DIR / "community_research.csv"

SHOT_FIELDS = [
    "date",
    "patch",
    "base",
    "release1",
    "release2",
    "speed",
    "player_3pt",
    "attempts",
    "makes",
    "greens",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries.

    Args:
        path: CSV path to read.

    Returns:
        Parsed rows, or an empty list when the file is missing or empty.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float.

    Args:
        value: Raw value to convert.
        default: Fallback value for blanks or invalid input.

    Returns:
        Converted float or the fallback.
    """
    try:
        if value in ("", None):
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Convert a value to a rounded integer.

    Args:
        value: Raw value to convert.
        default: Fallback value for blanks or invalid input.

    Returns:
        Rounded integer.
    """
    return int(round(_to_float(value, default)))


def _combo_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Build a normalized jumpshot combo key.

    Args:
        row: Row or payload containing base, release1, release2, and speed.

    Returns:
        Lowercased combo key tuple.
    """
    return (
        str(row.get("base", "")).strip().lower(),
        str(row.get("release1", "")).strip().lower(),
        str(row.get("release2", "")).strip().lower(),
        str(row.get("speed", "")).strip().lower(),
    )


def _clamp_pct(value: float) -> float:
    """Clamp a numeric value to the percentage range.

    Args:
        value: Percentage value to clamp.

    Returns:
        Value between 0.0 and 1.0.
    """
    return max(0.0, min(1.0, value))


def _parse_shot(row: dict[str, str]) -> dict[str, Any]:
    """Parse a tracked shot session row.

    Args:
        row: Raw CSV row from shots.csv.

    Returns:
        Normalized shot session with make and green percentages.
    """
    attempts = max(0, _to_int(row.get("attempts")))
    makes = max(0, min(attempts, _to_int(row.get("makes"))))
    greens = max(0, min(attempts, _to_int(row.get("greens"))))
    make_pct = makes / attempts if attempts else 0.0
    return {
        "date": row.get("date", ""),
        "patch": row.get("patch", ""),
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "player_3pt": _to_float(row.get("player_3pt")),
        "attempts": attempts,
        "makes": makes,
        "greens": greens,
        "make_pct": make_pct,
        "green_pct": greens / attempts if attempts else 0.0,
    }


def _parse_leaderboard(row: dict[str, str]) -> dict[str, Any]:
    """Parse a leaderboard row with backwards-compatible defaults.

    Args:
        row: Raw CSV row from exploit_leaderboard.csv.

    Returns:
        Normalized leaderboard row.
    """
    total_attempts = _to_int(row.get("total_attempts"))
    total_greens = _to_int(row.get("total_greens"))
    total_makes = _to_int(row.get("total_makes"))
    make_pct = _to_float(row.get("make_pct"))
    if not make_pct and total_makes and total_attempts:
        make_pct = total_makes / total_attempts
    green_pct = _to_float(row.get("green_pct"))
    if not green_pct and total_greens and total_attempts:
        green_pct = total_greens / total_attempts
    latency_score = _to_float(row.get("latency_score"))
    if not latency_score and make_pct > 0 and total_attempts:
        latency_score = (green_pct / make_pct) * math.log1p(total_attempts)

    parsed = {
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "total_attempts": total_attempts,
        "total_makes": total_makes,
        "make_pct": make_pct,
        "avg_residual": _to_float(row.get("avg_residual")),
        "avg_green_residual": _to_float(row.get("avg_green_residual")),
        "residual_std": _to_float(row.get("residual_std"), 0.0),
        "n_sessions": _to_int(row.get("n_sessions")),
        "total_greens": total_greens,
        "green_pct": green_pct,
        "exploit_score": _to_float(row.get("exploit_score")),
        "latency_score": latency_score,
    }
    return parsed


def _leaderboard_from_shots(
    shots: list[dict[str, Any]], leaderboard: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge saved leaderboard signals with current tracked shot volume.

    Args:
        shots: Parsed shot sessions.
        leaderboard: Parsed saved leaderboard rows.

    Returns:
        Leaderboard rows with current attempts, makes, greens, and latency score.
    """
    by_key = {_combo_key(row): dict(row) for row in leaderboard}
    shot_keys = sorted({_combo_key(row) for row in shots})

    for key in shot_keys:
        combo_shots = [row for row in shots if _combo_key(row) == key]
        attempts = sum(row["attempts"] for row in combo_shots)
        makes = sum(row["makes"] for row in combo_shots)
        greens = sum(row["greens"] for row in combo_shots)
        row = by_key.get(key)
        if row is None:
            first = combo_shots[0]
            row = {
                "base": first["base"],
                "release1": first["release1"],
                "release2": first["release2"],
                "speed": first["speed"],
                "avg_residual": 0.0,
                "avg_green_residual": 0.0,
                "residual_std": 0.0,
                "exploit_score": 0.0,
                "latency_score": 0.0,
            }
        row["total_attempts"] = attempts
        row["total_makes"] = makes
        row["make_pct"] = makes / attempts if attempts else 0.0
        row["n_sessions"] = len(combo_shots)
        row["total_greens"] = greens
        row["green_pct"] = greens / attempts if attempts else 0.0
        if not row.get("latency_score") and row["make_pct"] > 0:
            row["latency_score"] = (row["green_pct"] / row["make_pct"]) * math.log1p(attempts)
        by_key[key] = row

    rows = list(by_key.values())
    rows.sort(key=lambda row: row["exploit_score"], reverse=True)
    return rows


def _parse_patch_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse a patch impact row.

    Args:
        row: Raw patch impact or alert row.

    Returns:
        Normalized patch movement row.
    """
    return {
        "patch": row.get("patch", ""),
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "mean_make_pct": _to_float(row.get("mean_make_pct")),
        "mean_green_pct": _to_float(row.get("mean_green_pct")),
        "total_attempts": _to_int(row.get("total_attempts")),
        "avg_residual": _to_float(row.get("avg_residual")),
        "avg_green_residual": _to_float(row.get("avg_green_residual")),
        "total_greens": _to_int(row.get("total_greens")),
        "z_score": _to_float(row.get("z_score")),
    }


def _parse_research_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse a community or lab research row.

    Args:
        row: Raw row from community_research.csv.

    Returns:
        Normalized external research signal.
    """
    return {
        "source": row.get("source", ""),
        "source_url": row.get("source_url", ""),
        "source_quality": row.get("source_quality", ""),
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "player_3pt": _to_float(row.get("player_3pt")),
        "reported_make_pct": _to_float(row.get("reported_make_pct")),
        "reported_green_window_ms": _to_float(row.get("reported_green_window_ms")),
        "notes": row.get("notes", ""),
    }


def _weighted_make_pct(shots: list[dict[str, Any]]) -> float | None:
    """Compute a weighted make percentage.

    Args:
        shots: Parsed shot sessions.

    Returns:
        Attempts-weighted make percentage, or None when no attempts exist.
    """
    attempts = sum(row["attempts"] for row in shots)
    if attempts <= 0:
        return None
    return sum(row["makes"] for row in shots) / attempts


def _baseline_make_pct(shots: list[dict[str, Any]], player_3pt: float) -> float:
    """Estimate a baseline make percentage by 3PT rating.

    Args:
        shots: Parsed shot sessions.
        player_3pt: Player three-point rating.

    Returns:
        Clamped baseline make percentage.
    """
    weighted = _weighted_make_pct(shots)
    if weighted is None:
        return 0.6

    total_weight = sum(row["attempts"] for row in shots)
    if total_weight <= 0:
        return weighted

    mean_x = sum(row["player_3pt"] * row["attempts"] for row in shots) / total_weight
    mean_y = sum(row["make_pct"] * row["attempts"] for row in shots) / total_weight
    numerator = sum(
        row["attempts"] * (row["player_3pt"] - mean_x) * (row["make_pct"] - mean_y)
        for row in shots
    )
    denominator = sum(row["attempts"] * (row["player_3pt"] - mean_x) ** 2 for row in shots)
    if denominator == 0:
        return weighted
    slope = numerator / denominator
    return _clamp_pct(mean_y + slope * (player_3pt - mean_x))


def _recommendation(score: float, attempts: int) -> str:
    """Choose a simple human-readable jumpshot recommendation.

    Args:
        score: Exploit score from the leaderboard.
        attempts: Total attempts for the combo.

    Returns:
        Recommendation text.
    """
    if attempts < 100:
        return "Needs more test volume"
    if score >= 0.10:
        return "Meta candidate"
    if score >= 0.03:
        return "Playable edge"
    if score <= -0.10:
        return "Avoid for now"
    return "Stable baseline"


def _confidence(attempts: int, sessions: int) -> str:
    """Classify confidence from attempts and session count.

    Args:
        attempts: Total attempts for a combo.
        sessions: Number of tracked sessions for a combo.

    Returns:
        One of low, medium, or high.
    """
    if attempts >= 300 and sessions >= 3:
        return "high"
    if attempts >= 150 and sessions >= 2:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def load_state() -> dict[str, Any]:
    """Load CSV-backed app state once and cache it.

    Returns:
        Parsed shots, leaderboard, patch data, research, and latest patch.
    """
    shots = [_parse_shot(row) for row in _read_csv(SHOTS_PATH)]
    saved_leaderboard = [_parse_leaderboard(row) for row in _read_csv(LEADERBOARD_PATH)]
    leaderboard = _leaderboard_from_shots(shots, saved_leaderboard)
    patch_impact = [_parse_patch_row(row) for row in _read_csv(PATCH_IMPACT_PATH)]
    patch_alerts = [_parse_patch_row(row) for row in _read_csv(PATCH_ALERTS_PATH)]
    community_research = [_parse_research_row(row) for row in _read_csv(COMMUNITY_RESEARCH_PATH)]

    latest_patch = max((row["patch"] for row in shots), default="")
    return {
        "shots": shots,
        "leaderboard": leaderboard,
        "patch_impact": patch_impact,
        "patch_alerts": patch_alerts,
        "community_research": community_research,
        "latest_patch": latest_patch,
    }


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    """Load saved model artifacts once when dependencies are available.

    Returns:
        Dict containing optional make and green model objects.
    """
    try:
        import joblib
    except ImportError:
        return {"make_model": None, "green_model": None}

    def load_optional_model(path: Path) -> Any:
        """Load one model artifact if its dependencies are available.

        Args:
            path: Pickle artifact path.

        Returns:
            Loaded model, or None when missing or unloadable.
        """
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    return {
        "make_model": load_optional_model(MODEL_PATH),
        "green_model": load_optional_model(GREEN_MODEL_PATH),
    }


def get_summary() -> dict[str, Any]:
    """Return summary data for the frontend.

    Returns:
        App summary, select options, leaderboard, patch data, and research signals.
    """
    state = load_state()
    shots = state["shots"]
    leaderboard = state["leaderboard"]
    total_attempts = sum(row["attempts"] for row in shots)
    total_makes = sum(row["makes"] for row in shots)
    total_greens = sum(row["greens"] for row in shots)

    options = {
        "bases": sorted({row["base"] for row in shots if row["base"]}),
        "release1": sorted({row["release1"] for row in shots if row["release1"]}),
        "release2": sorted({row["release2"] for row in shots if row["release2"]}),
        "speeds": sorted({row["speed"] for row in shots if row["speed"]}),
        "patches": sorted({row["patch"] for row in shots if row["patch"]}),
    }

    return {
        "summary": {
            "sessions": len(shots),
            "total_attempts": total_attempts,
            "make_pct": total_makes / total_attempts if total_attempts else 0.0,
            "green_pct": total_greens / total_attempts if total_attempts else 0.0,
            "latest_patch": state["latest_patch"],
            "top_score": leaderboard[0]["exploit_score"] if leaderboard else 0.0,
            "top_latency_score": leaderboard[0]["latency_score"] if leaderboard else 0.0,
            "green_model_available": load_models()["green_model"] is not None,
        },
        "options": options,
        "leaderboard": leaderboard,
        "patch_impact": state["patch_impact"],
        "patch_alerts": state["patch_alerts"],
        "community_research": state["community_research"],
    }


def score_jumpshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a jumpshot payload against tracked data and leaderboard signals.

    Args:
        payload: Jumpshot build payload.

    Returns:
        Prediction, exploit, latency, confidence, and recent session data.
    """
    state = load_state()
    shots = state["shots"]
    key = _combo_key(payload)
    player_3pt = _to_float(payload.get("player_3pt"), 92.0)

    exact_sessions = [row for row in shots if _combo_key(row) == key]
    exact = next((row for row in state["leaderboard"] if _combo_key(row) == key), None)
    attempts = sum(row["attempts"] for row in exact_sessions)
    makes = sum(row["makes"] for row in exact_sessions)
    greens = sum(row["greens"] for row in exact_sessions)
    actual_make_pct = makes / attempts if attempts else None
    actual_green_pct = greens / attempts if attempts else None

    baseline = _baseline_make_pct(shots, player_3pt)
    if exact and actual_make_pct is not None:
        expected_make_pct = _clamp_pct(actual_make_pct - exact["avg_residual"])
        expected_green_pct = _clamp_pct((actual_green_pct or 0.0) - exact["avg_green_residual"])
        edge = exact["avg_residual"]
        green_edge = exact["avg_green_residual"]
        exploit_score = exact["exploit_score"]
        latency_score = exact["latency_score"]
        sessions = exact["n_sessions"]
    else:
        expected_make_pct = baseline
        expected_green_pct = None
        edge = 0.0
        green_edge = 0.0
        exploit_score = 0.0
        latency_score = 0.0
        sessions = len(exact_sessions)

    recent = sorted(exact_sessions, key=lambda row: row["date"], reverse=True)[:5]
    return {
        "input": {
            "base": payload.get("base", ""),
            "release1": payload.get("release1", ""),
            "release2": payload.get("release2", ""),
            "speed": payload.get("speed", ""),
            "player_3pt": player_3pt,
        },
        "matched": exact is not None,
        "actual_make_pct": actual_make_pct,
        "expected_make_pct": expected_make_pct,
        "expected_green_pct": expected_green_pct,
        "baseline_make_pct": baseline,
        "edge": edge,
        "green_edge": green_edge,
        "exploit_score": exploit_score,
        "latency_score": latency_score,
        "attempts": attempts,
        "makes": makes,
        "greens": greens,
        "green_pct": actual_green_pct,
        "sessions": sessions,
        "confidence": _confidence(attempts, sessions),
        "recommendation": _recommendation(exploit_score, attempts),
        "recent_sessions": recent,
    }


def add_session(payload: dict[str, Any]) -> dict[str, Any]:
    """Append a real tracked shooting session to shots.csv.

    Args:
        payload: Validated session payload.

    Returns:
        Saved session plus updated score response.
    """
    attempts = _to_int(payload.get("attempts"))
    makes = _to_int(payload.get("makes"))
    player_3pt = _to_float(payload.get("player_3pt"), 92.0)
    if attempts <= 0:
        raise ValueError("attempts must be greater than zero")
    if makes < 0 or makes > attempts:
        raise ValueError("makes must be between zero and attempts")

    greens_value = payload.get("greens")
    if greens_value in ("", None):
        raise ValueError("greens is required - count your greens each session")
    greens = _to_int(greens_value)
    if greens < 0 or greens > attempts:
        raise ValueError("greens must be between zero and attempts")

    row = {
        "date": str(payload.get("date") or date.today().isoformat()),
        "patch": str(payload.get("patch") or load_state()["latest_patch"] or "1.00").strip(),
        "base": str(payload.get("base", "")).strip(),
        "release1": str(payload.get("release1", "")).strip(),
        "release2": str(payload.get("release2", "")).strip(),
        "speed": str(payload.get("speed", "")).strip(),
        "player_3pt": f"{player_3pt:g}",
        "attempts": str(attempts),
        "makes": str(makes),
        "greens": str(greens),
    }
    missing = [field for field in ("base", "release1", "release2", "speed") if not row[field]]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    file_exists = SHOTS_PATH.exists() and SHOTS_PATH.stat().st_size > 0
    with SHOTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    load_state.cache_clear()
    parsed = _parse_shot(row)
    return {
        "session": parsed,
        "score": score_jumpshot(parsed),
    }
