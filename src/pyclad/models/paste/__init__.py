from pyclad.models.paste.config import PaSTeConfig

__all__ = ["PaSTe", "PaSTeConfig"]


def __getattr__(name: str):
    if name == "PaSTe":
        from pyclad.models.paste.paste import PaSTe

        return PaSTe
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
