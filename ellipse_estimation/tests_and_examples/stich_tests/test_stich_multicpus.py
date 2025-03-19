#!/usr/bin/env python
# coding: utf-8

# In[1]:
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:
import iris
import iris.quickplot as qplt
from iris.fileformats import netcdf as inc
from iris.coords import DimCoord, AuxCoord
from iris.cube import Cube
import matplotlib.pyplot as plt
import numpy as np


# In[3]:
import os
import multiprocessing

print(multiprocessing.cpu_count())
use_joblib = True
# speed: serial > threading > loky
# backend = 'loky'
backend = "threading"
# inplace = True
inplace = False
n_jobs = 4


# In[4]:


import cProfile
import pstats


# In[5]:
import socket

print(socket.gethostname())


# In[6]:
from nonstationary_cov import (
    cube_covariance_nonstationary_stich as cube_cov_stich,
)


# In[7]:
if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [15, 10]

    # In[8]:

    cube_matern_dist = iris.load("test_data/Atlantic_Ocean_07.nc")
    print(repr(cube_matern_dist))

    # In[9]:

    Lx = cube_matern_dist.extract("Lx")[0][50:70, 50:70]
    Ly = cube_matern_dist.extract("Ly")[0][50:70, 50:70]
    theta = cube_matern_dist.extract("theta")[0][50:70, 50:70]
    sigma = cube_matern_dist.extract("standard_deviation")[0][50:70, 50:70]

    # In[10]:

    print(repr(Lx))
    print(repr(Ly))
    print(repr(theta))
    print(repr(sigma))

    # In[11]:

    print(Lx)
    v = float(Lx.coord("v_shape").points)
    delta_x_method = "Modified_Met_Office"
    print(v)
    print(delta_x_method)

    # In[12]:

    print(Lx.coord("latitude"))
    print(Lx.coord("longitude"))

    # In[13]:

    print(os.cpu_count())
    profiler = cProfile.Profile()
    profiler.enable()
    hfix_cpd = True
    stich_hfix = cube_cov_stich.CovarianceCube_PreStichedLocalEstimates(
        Lx,
        Ly,
        theta,
        sigma,
        v=v,
        delta_x_method=delta_x_method,
        check_positive_definite=hfix_cpd,
        use_joblib=use_joblib,
        n_jobs=n_jobs,
        backend=backend,
        inplace=inplace,
        nolazy=True,
    )
    profiler.disable()

    # In[14]:

    print("tottime")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(15)

    print("cumtime")
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(15)

    # In[15]:

    print(stich_hfix)

    # In[16]:

    xy = 150
    cor_with_point_cube0 = stich_hfix.remap_one_point_2_map(
        stich_hfix.cor_ns[xy, :].copy(), cube_name="correlation", cube_unit="1"
    )

    # In[17]:

    qplt.contourf(cor_with_point_cube0)

    # In[18]:

    stich = stich_hfix
    print(hfix_cpd)
    print(stich.check_positive_definite)
    print(int(stich.check_positive_definite))
    covariance_file = "test_hfix.nc"

    # In[19]:

    stich_cov = stich.cov_ns
    stich_cor = stich.cor_ns
    # Create dummy row and column coordinates for covariance and correlation matrix
    nrows = stich.cov_ns.shape[0]
    dim_row = DimCoord(
        np.arange(nrows, dtype=int), long_name="dim_0", units="1"
    )
    dim_col = DimCoord(
        np.arange(nrows, dtype=int), long_name="dim_1", units="1"
    )
    # v_coord as an Aux Coord stating the Matern parameter
    v_coord = AuxCoord(v, long_name="matern_nu")
    det_coord = AuxCoord(stich.cov_det, long_name="covariance_determinant")
    eig_coord = AuxCoord(stich.cov_eig[-1], long_name="smallest_eigenvalue")
    pd_check_coord = AuxCoord(
        int(stich.check_positive_definite),
        long_name="positive_semidefinite_check_enabled",
    )
    # Define the iris cube
    cov_cube = Cube(stich_cov, dim_coords_and_dims=[(dim_row, 0), (dim_col, 1)])
    cor_cube = Cube(stich_cor, dim_coords_and_dims=[(dim_row, 0), (dim_col, 1)])
    cov_cube.data = cov_cube.data.astype(np.float32)
    cor_cube.data = cor_cube.data.astype(np.float32)
    cov_cube.add_aux_coord(v_coord)
    cor_cube.add_aux_coord(v_coord)
    cov_cube.add_aux_coord(det_coord)
    cor_cube.add_aux_coord(det_coord)
    cov_cube.add_aux_coord(eig_coord)
    cor_cube.add_aux_coord(eig_coord)
    cov_cube.add_aux_coord(pd_check_coord)
    cor_cube.add_aux_coord(pd_check_coord)
    cov_cube.units = "K**2"
    cor_cube.units = "1"
    cov_cube.rename("covariance")
    cor_cube.rename("correlation")

    # Write to file
    cov_list = iris.cube.CubeList()
    cov_list.append(cov_cube)
    cov_list.append(cor_cube)
    print("Writing covariance file: ", covariance_file)
    inc.save(cov_list, covariance_file)
    ##
