"""Types and Literals used by GloMarGridding functions and methods."""

from typing import Literal

ModelType = Literal[
    "ps2006_kks2011_iso",
    "ps2006_kks2011_ani",
    "ps2006_kks2011_ani_r",
    "ps2006_kks2011_iso_pd",
    "ps2006_kks2011_ani_pd",
    "ps2006_kks2011_ani_r_pd",
]

FForm = Literal[
    "anisotropic_rotated",
    "anisotropic",
    "isotropic",
    "anisotropic_rotated_pd",
    "anisotropic_pd",
    "isotropic_pd",
]

SuperCategory = Literal[
    "1_param_matern",
    "2_param_matern",
    "3_param_matern",
    "1_param_matern_pd",
    "2_param_matern_pd",
    "3_param_matern_pd",
]

DeltaXMethod = Literal["Met_Office", "Modified_Met_Office"]

CovarianceMethod = Literal["batched", "low_memory", "array"]
