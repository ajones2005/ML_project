from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from jumpshot_predictor import meta_detector


def test_build_exploit_leaderboard_adds_latency_score() -> None:
    """Calculate green percentage and latency score from real green counts."""
    shots = pd.DataFrame(
        [
            {
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "date": "2026-01-01",
                "attempts": 100,
                "makes": 70,
                "greens": 49,
                "residual": 0.02,
                "green_residual": 0.04,
            },
            {
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "date": "2026-01-02",
                "attempts": 100,
                "makes": 66,
                "greens": 41,
                "residual": 0.01,
                "green_residual": 0.02,
            },
        ]
    )

    leaderboard = meta_detector.build_exploit_leaderboard(shots)
    row = leaderboard.iloc[0]

    assert row["make_pct"] == pytest.approx(0.68)
    assert row["green_pct"] == pytest.approx(0.45)
    assert row["avg_green_residual"] == pytest.approx(0.03)
    assert row["latency_score"] == pytest.approx((0.45 / 0.68) * np.log1p(200))


def test_train_baseline_rejects_unknown_target() -> None:
    """Raise a KeyError when a caller requests an unknown target column."""
    shots = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "base": "Curry",
                "release1": "Kobe",
                "release2": "Gay",
                "speed": "max",
                "patch": "1.06",
                "player_3pt": 92,
                "attempts": 100,
                "make_pct": 0.68,
            }
        ]
    )

    with pytest.raises(KeyError):
        meta_detector.train_baseline(shots, target="missing")
