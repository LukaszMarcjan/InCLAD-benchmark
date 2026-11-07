from pyclad.models.draem.config import DRAEMConfig
from pyclad.models.draem.draem import build_architecture


def build(config: DRAEMConfig):
    return build_architecture(config)
