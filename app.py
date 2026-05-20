from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory
from pydantic import BaseModel, Field, ValidationError

from jumpshot_predictor.service import add_session, get_summary, load_models, load_state, score_jumpshot


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
CHART_PATH = ROOT / "jumpshot_predictor" / "exploit_chart.png"


class ScoreRequest(BaseModel):
    """Validate a jumpshot score request."""

    base: str = Field(..., min_length=1)
    release1: str = Field(..., min_length=1)
    release2: str = Field(..., min_length=1)
    speed: str = Field(..., min_length=1)
    player_3pt: float = Field(92.0, ge=25, le=99)


class SessionRequest(ScoreRequest):
    """Validate a tracked jumpshot test session request."""

    attempts: int = Field(..., gt=0)
    makes: int = Field(..., ge=0)
    greens: int = Field(..., ge=0)
    date: Optional[str] = None
    patch: Optional[str] = None


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a Pydantic model as a plain dict across Pydantic versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _validate_payload(schema: type[BaseModel]) -> dict[str, Any]:
    """Validate request JSON against a Pydantic schema and return a dict."""
    payload = request.get_json(silent=True) or {}
    if hasattr(schema, "model_validate"):
        model = schema.model_validate(payload)
    else:
        model = schema.parse_obj(payload)
    return _model_to_dict(model)


def _json_error(message: str, status: int) -> tuple[Any, int]:
    """Build a consistent JSON error response."""
    return jsonify({"error": message}), status


def create_app() -> Flask:
    """Create and configure the NBA 2K26 jumpshot creator Flask app."""
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

    load_state()
    load_models()

    @app.get("/api/health")
    def health() -> Any:
        """Return a health check response."""
        return jsonify({"ok": True})

    @app.get("/api/summary")
    def summary() -> Any:
        """Return current jumpshot summary, leaderboard, and patch data."""
        return jsonify(get_summary())

    @app.post("/api/jumpshot/score")
    def score() -> Any:
        """Score a jumpshot build from validated request data."""
        payload = _validate_payload(ScoreRequest)
        return jsonify(score_jumpshot(payload))

    @app.post("/api/sessions")
    def sessions() -> Any:
        """Save a tracked test session and return its updated score."""
        payload = _validate_payload(SessionRequest)
        return jsonify(add_session(payload)), 201

    @app.get("/jumpshot_predictor/exploit_chart.png")
    def exploit_chart() -> Any:
        """Serve the generated exploit chart image."""
        return send_from_directory(CHART_PATH.parent, CHART_PATH.name)

    @app.get("/")
    def index() -> Any:
        """Serve the frontend entry point."""
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.errorhandler(ValueError)
    def handle_value_error(exc: ValueError) -> tuple[Any, int]:
        """Return service validation failures as JSON."""
        return _json_error(str(exc), 400)

    @app.errorhandler(ValidationError)
    def handle_validation_error(exc: ValidationError) -> tuple[Any, int]:
        """Return Pydantic validation failures as JSON."""
        return _json_error(str(exc.errors()[0]["msg"]), 400)

    @app.errorhandler(404)
    def handle_not_found(_: Exception) -> tuple[Any, int]:
        """Return unknown routes as JSON errors for API paths."""
        if request.path.startswith("/api/"):
            return _json_error("not found", 404)
        return _json_error("not found", 404)

    return app


def run(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Run the Flask development server."""
    create_app().run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
