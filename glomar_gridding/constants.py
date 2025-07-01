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

"""Constants used by various functions and methods within the library"""

RADIUS_OF_EARTH_M: float = 6371000.0  # Average radius of Earth (m)
RADIUS_OF_EARTH_KM: float = 6371.0  # Average radius of Earth (m)
KM_TO_M: float = 1000.0

# Each degree of latitude is equal to 60 nautical miles (with cosine correction
# for lon values)
NM_PER_LAT: float = 60.0  # 60 nautical miles per degree latitude
KM_TO_NM: float = 1.852  # 1852 meters per nautical miles

DEFAULT_N_JOBS: int = 4
DEFAULT_BACKEND: str = "loky"  # loky appears to be fastest
