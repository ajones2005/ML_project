from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jumpshot_predictor import service


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a CSV file.

    Args:
        path: File path to write.
        rows: CSV rows to write.

    Returns:
        None.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the service module at isolated CSV files.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Temporary data directory path.
    """
    shots = tmp_path / "shots.csv"
    leaderboard = tmp_path / "exploit_leaderboard.csv"
    patch_impact = tmp_path / "patch_impact.csv"
    patch_alerts = tmp_path / "patch_alerts.csv"
    research = tmp_path / "community_research.csv"

    write_csv(
        shots,
        [
            {
                "date": "2026-01-01",
                "patch": "1.06",
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "player_3pt": 92,
                "attempts": 100,
                "makes": 68,
                "greens": 45,
            }
        ],
    )
    write_csv(
        leaderboard,
        [
            {
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "total_attempts": 100,
                "total_makes": 68,
                "make_pct": 0.68,
                "avg_residual": 0.02,
                "avg_green_residual": 0.03,
                "residual_std": 0.0,
                "n_sessions": 1,
                "total_greens": 45,
                "green_pct": 0.45,
                "exploit_score": 0.12,
                "latency_score": 3.05,
            }
        ],
    )
    write_csv(
        patch_impact,
        [
            {
                "patch": "1.06",
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "mean_make_pct": 0.68,
                "mean_green_pct": 0.45,
                "total_attempts": 100,
                "avg_residual": 0.02,
                "avg_green_residual": 0.03,
                "total_greens": 45,
                "z_score": 0,
            }
        ],
    )
    patch_alerts.write_text(
        "patch,base,release1,release2,speed,mean_make_pct,mean_green_pct,total_attempts,avg_residual,avg_green_residual,total_greens,z_score\n",
        encoding="utf-8",
    )
    research.write_text(
        "source,source_url,source_quality,base,release1,release2,speed,player_3pt,reported_make_pct,reported_green_window_ms,notes\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(service, "SHOTS_PATH", shots)
    monkeypatch.setattr(service, "LEADERBOARD_PATH", leaderboard)
    monkeypatch.setattr(service, "PATCH_IMPACT_PATH", patch_impact)
    monkeypatch.setattr(service, "PATCH_ALERTS_PATH", patch_alerts)
    monkeypatch.setattr(service, "COMMUNITY_RESEARCH_PATH", research)
    monkeypatch.setattr(service, "MODEL_PATH", tmp_path / "missing_make.pkl")
    monkeypatch.setattr(service, "GREEN_MODEL_PATH", tmp_path / "missing_green.pkl")
    service.load_state.cache_clear()
    service.load_models.cache_clear()
    return tmp_path


def test_add_session_requires_real_greens(isolated_data: Path) -> None:
    """Require manually tracked greens when adding a session."""
    with pytest.raises(ValueError, match="greens is required"):
        service.add_session(
            {
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "player_3pt": 92,
                "attempts": 100,
                "makes": 68,
            }
        )


def test_score_jumpshot_returns_latency_fields(isolated_data: Path) -> None:
    """Return latency and green residual fields in score responses."""
    scored = service.score_jumpshot(
        {
            "base": "Curry",
            "release1": "Kobe",
            "release2": "Gay",
            "speed": "max",
            "player_3pt": 92,
        }
    )

    assert scored["latency_score"] == pytest.approx(3.05)
    assert scored["green_edge"] == pytest.approx(0.03)
    assert scored["expected_green_pct"] == pytest.approx(0.42)


def test_add_session_appends_tracked_greens(isolated_data: Path) -> None:
    """Append a session using the provided green count."""
    result = service.add_session(
        {
            "base": "Curry",
            "release1": "Kobe",
            "release2": "Gay",
            "speed": "max",
            "player_3pt": 92,
            "attempts": 50,
            "makes": 35,
            "greens": 25,
            "patch": "1.07",
            "date": "2026-02-01",
        }
    )

    assert result["session"]["greens"] == 25
    assert service.get_summary()["summary"]["total_attempts"] == 150
