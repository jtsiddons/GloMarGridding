# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
