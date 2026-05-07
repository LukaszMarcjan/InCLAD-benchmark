from pyclad.models.fastflow.config import FastFlowConfig

__all__ = ["FastFlow", "FastFlowConfig"]


def __getattr__(name: str):
    if name == "FastFlow":
        from pyclad.models.fastflow.fastflow import FastFlow

        return FastFlow
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
