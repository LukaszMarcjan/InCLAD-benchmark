from pyclad.models.cfa.config import CFAConfig

__all__ = ["CFA", "CFAConfig"]


def __getattr__(name: str):
    if name == "CFA":
        from pyclad.models.cfa.cfa import CFA

        return CFA
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
