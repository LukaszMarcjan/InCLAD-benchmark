__all__ = ["PaDiMCLStrategy", "PatchCoreCLStrategy", "CFACLStrategy"]


def __getattr__(name: str):
    if name == "PaDiMCLStrategy":
        from pyclad.strategies.vision.padim_cl import PaDiMCLStrategy

        return PaDiMCLStrategy
    if name == "PatchCoreCLStrategy":
        from pyclad.strategies.vision.patchcore_cl import PatchCoreCLStrategy

        return PatchCoreCLStrategy
    if name == "CFACLStrategy":
        from pyclad.strategies.vision.cfa_cl import CFACLStrategy

        return CFACLStrategy
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
