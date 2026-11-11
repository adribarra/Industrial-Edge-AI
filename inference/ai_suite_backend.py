"""Placeholder for the Altera FPGA AI Suite backend (a later phase).

It is not implemented in Phase 1. Nothing runs on an FPGA and nothing is simulated.
"""
from inference.base import BackendError, InferenceBackend, Prediction


class AISuiteBackend(InferenceBackend):
    """Empty backend that only exists so the app is ready for the AI Suite later."""

    name = "ai_suite"
    device = "fpga"

    def load_model(self, model_path) -> None:
        """Always fails for now, since the AI Suite part is not built yet."""
        raise BackendError("AI Suite backend is not implemented (planned for a later phase)")

    def predict(self, image) -> Prediction:
        """Always fails for now, since the AI Suite part is not built yet."""
        raise BackendError("AI Suite backend is not implemented (planned for a later phase)")
