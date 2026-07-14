from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.ml.train import MODEL_FILENAME


class PredictionResult:
    __slots__ = ("predicted_label", "confidence", "top_predictions", "needs_review")

    def __init__(
        self,
        predicted_label: str,
        confidence: float,
        top_predictions: list[dict],
        needs_review: bool,
    ) -> None:
        self.predicted_label = predicted_label
        self.confidence = confidence
        self.top_predictions = top_predictions
        self.needs_review = needs_review


class TicketPredictor:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self._pipeline: Pipeline | None = None
        self._model_path = model_path
        self._threshold = confidence_threshold

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self._model_path}. Run `python -m app.ml.train` first."
            )
        self._pipeline = joblib.load(self._model_path)

    def predict(self, text: str) -> PredictionResult:
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        probabilities = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_

        ranked = sorted(zip(classes, probabilities), key=lambda p: p[1], reverse=True)
        top_two = [
            {"label": label, "confidence": round(float(prob), 4)}
            for label, prob in ranked[:2]
        ]
        confidence = top_two[0]["confidence"]

        return PredictionResult(
            predicted_label=top_two[0]["label"],
            confidence=confidence,
            top_predictions=top_two,
            needs_review=confidence < self._threshold,
        )


_predictor: TicketPredictor | None = None


def get_predictor() -> TicketPredictor:
    global _predictor
    if _predictor is None:
        settings = get_settings()
        _predictor = TicketPredictor(
            model_path=settings.artifacts_dir / MODEL_FILENAME,
            confidence_threshold=settings.confidence_threshold,
        )
    return _predictor
