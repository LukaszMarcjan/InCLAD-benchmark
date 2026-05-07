from pyclad.models.ganomaly.config import GANomalyConfig

__all__ = ["GANomaly", "GANomalyConfig"]


def __getattr__(name: str):
    if name == "GANomaly":
        from pyclad.models.ganomaly.ganomaly import GANomaly

        return GANomaly
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
