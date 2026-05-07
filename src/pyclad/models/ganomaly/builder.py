from pyclad.models.ganomaly.architecture import GANomalyArchitecture
from pyclad.models.ganomaly.config import GANomalyConfig


def build(config: GANomalyConfig) -> GANomalyArchitecture:
    return GANomalyArchitecture(
        input_size=config.input_size,
        in_channels=config.in_channels,
        n_features=config.n_features,
        latent_vec_size=config.latent_vec_size,
        extra_layers=config.extra_layers,
        add_final_conv_layer=config.add_final_conv_layer,
    )
