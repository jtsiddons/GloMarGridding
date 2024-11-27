import netCDF4 as nc
import numpy as np


def add_empty_layers(nc_variables: list[nc.Variable] | nc.Variable,
                     timestamps: list[int] | int,
                     shape: tuple[int, int],
                     ) -> None:
    empty = np.zeros(shape=shape).astype(np.float32)
    nc_variables = [nc_variables] if not isinstance(nc_variables, list) else nc_variables
    timestamps = [timestamps] if not isinstance(timestamps, list) else timestamps
    for variable in nc_variables:
        for timestamp in timestamps:
            variable[timestamp, :, :] = empty
    return None
