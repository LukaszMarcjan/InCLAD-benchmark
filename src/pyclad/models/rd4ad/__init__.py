from pyclad.models.rd4ad.config import RD4ADConfig

__all__ = ["RD4AD", "RD4ADConfig"]


def __getattr__(name: str):
    if name == "RD4AD":
        from pyclad.models.rd4ad.rd4ad import RD4AD

        return RD4AD
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
