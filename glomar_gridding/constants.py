"""Constants used by various functions and methods within the library"""

RADIUS_OF_EARTH_M: float = 6371000.0  # Average radius of Earth (m)
RADIUS_OF_EARTH_KM: float = 6371.0  # Average radius of Earth (m)
KM_TO_M: float = 1000.0

# Each degree of latitude is equal to 60 nautical miles (with cosine correction
# for lon values)
NM_PER_LAT: float = 60.0  # 60 nautical miles per degree latitude
KM_TO_NM: float = 1.852  # 1852 meters per nautical miles
