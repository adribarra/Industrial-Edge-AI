"""The inference backends. Use create_backend() so the app never imports a ML library directly."""
from inference.base import BackendError, InferenceBackend, Prediction


def create_backend(name: str) -> InferenceBackend:
    """Return the backend with the given name ("cpu" or "ai_suite")."""
    if name == "cpu":
        from inference.cpu_backend import CPUBackend  # imported here because it loads PyTorch
        return CPUBackend()
    if name == "ai_suite":
        from inference.ai_suite_backend import AISuiteBackend
        return AISuiteBackend()
    raise ValueError(f"Unknown backend: {name!r}")


__all__ = ["BackendError", "InferenceBackend", "Prediction", "create_backend"]
