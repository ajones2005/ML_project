from __future__ import annotations

import csv
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
SHOTS_PATH = DATA_DIR / "shots.csv"
LEADERBOARD_PATH = DATA_DIR / "exploit_leaderboard.csv"
PATCH_IMPACT_PATH = DATA_DIR / "patch_impact.csv"
PATCH_ALERTS_PATH = DATA_DIR / "patch_alerts.csv"

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
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any, default: float = 0.0) -> float:
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
    return int(round(_to_float(value, default)))


def _combo_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("base", "")).strip().lower(),
        str(row.get("release1", "")).strip().lower(),
        str(row.get("release2", "")).strip().lower(),
        str(row.get("speed", "")).strip().lower(),
    )


def _clamp_pct(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_shot(row: dict[str, str]) -> dict[str, Any]:
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
    parsed = {
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "total_attempts": _to_int(row.get("total_attempts")),
        "avg_residual": _to_float(row.get("avg_residual")),
        "residual_std": _to_float(row.get("residual_std"), 0.0),
        "n_sessions": _to_int(row.get("n_sessions")),
        "total_greens": _to_int(row.get("total_greens")),
        "exploit_score": _to_float(row.get("exploit_score")),
    }
    parsed["green_pct"] = (
        parsed["total_greens"] / parsed["total_attempts"]
        if parsed["total_attempts"]
        else 0.0
    )
    return parsed


def _leaderboard_from_shots(
    shots: list[dict[str, Any]], leaderboard: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {_combo_key(row): dict(row) for row in leaderboard}
    shot_keys = sorted({_combo_key(row) for row in shots})

    for key in shot_keys:
        combo_shots = [row for row in shots if _combo_key(row) == key]
        attempts = sum(row["attempts"] for row in combo_shots)
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
                "residual_std": 0.0,
                "exploit_score": 0.0,
            }
        row["total_attempts"] = attempts
        row["n_sessions"] = len(combo_shots)
        row["total_greens"] = greens
        row["green_pct"] = greens / attempts if attempts else 0.0
        by_key[key] = row

    rows = list(by_key.values())
    rows.sort(key=lambda row: row["exploit_score"], reverse=True)
    return rows


def _parse_patch_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "patch": row.get("patch", ""),
        "base": row.get("base", ""),
        "release1": row.get("release1", ""),
        "release2": row.get("release2", ""),
        "speed": row.get("speed", ""),
        "mean_make_pct": _to_float(row.get("mean_make_pct")),
        "total_attempts": _to_int(row.get("total_attempts")),
        "avg_residual": _to_float(row.get("avg_residual")),
        "total_greens": _to_int(row.get("total_greens")),
        "z_score": _to_float(row.get("z_score")),
    }


def _weighted_make_pct(shots: list[dict[str, Any]]) -> float | None:
    attempts = sum(row["attempts"] for row in shots)
    if attempts <= 0:
        return None
    return sum(row["makes"] for row in shots) / attempts


def _baseline_make_pct(shots: list[dict[str, Any]], player_3pt: float) -> float:
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
    if attempts >= 300 and sessions >= 3:
        return "high"
    if attempts >= 150 and sessions >= 2:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def load_state() -> dict[str, Any]:
    shots = [_parse_shot(row) for row in _read_csv(SHOTS_PATH)]
    saved_leaderboard = [_parse_leaderboard(row) for row in _read_csv(LEADERBOARD_PATH)]
    leaderboard = _leaderboard_from_shots(shots, saved_leaderboard)
    patch_impact = [_parse_patch_row(row) for row in _read_csv(PATCH_IMPACT_PATH)]
    patch_alerts = [_parse_patch_row(row) for row in _read_csv(PATCH_ALERTS_PATH)]

    latest_patch = max((row["patch"] for row in shots), default="")
    return {
        "shots": shots,
        "leaderboard": leaderboard,
        "patch_impact": patch_impact,
        "patch_alerts": patch_alerts,
        "latest_patch": latest_patch,
    }


def get_summary() -> dict[str, Any]:
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
        },
        "options": options,
        "leaderboard": leaderboard,
        "patch_impact": state["patch_impact"],
        "patch_alerts": state["patch_alerts"],
    }


def score_jumpshot(payload: dict[str, Any]) -> dict[str, Any]:
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

    baseline = _baseline_make_pct(shots, player_3pt)
    if exact and actual_make_pct is not None:
        expected_make_pct = _clamp_pct(actual_make_pct - exact["avg_residual"])
        edge = exact["avg_residual"]
        exploit_score = exact["exploit_score"]
        sessions = exact["n_sessions"]
    else:
        expected_make_pct = baseline
        edge = 0.0
        exploit_score = 0.0
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
        "baseline_make_pct": baseline,
        "edge": edge,
        "exploit_score": exploit_score,
        "attempts": attempts,
        "makes": makes,
        "greens": greens,
        "green_pct": greens / attempts if attempts else None,
        "sessions": sessions,
        "confidence": _confidence(attempts, sessions),
        "recommendation": _recommendation(exploit_score, attempts),
        "recent_sessions": recent,
    }


def add_session(payload: dict[str, Any]) -> dict[str, Any]:
    attempts = _to_int(payload.get("attempts"))
    makes = _to_int(payload.get("makes"))
    player_3pt = _to_float(payload.get("player_3pt"), 92.0)
    if attempts <= 0:
        raise ValueError("attempts must be greater than zero")
    if makes < 0 or makes > attempts:
        raise ValueError("makes must be between zero and attempts")

    greens_value = payload.get("greens")
    if greens_value in ("", None):
        make_pct = makes / attempts
        green_factor = 0.65 + (player_3pt - 85) * 0.003
        greens = round(make_pct * green_factor * attempts)
    else:
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
