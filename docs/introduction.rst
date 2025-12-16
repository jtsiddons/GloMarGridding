============
Introduction
============

Global surface temperature datasets - such as those used in the Intergovernmental Panel on Climate
Change (IPCC) Assessment Report [IPCC]_ rely on a range of techniques to generate smoothed, infilled
gridded fields from available observations [Lenssen]_, [Kadow]_, [RohdeBerkeley]_, [Morice_2021]_,
[Huang]_ . The construction of these datasets involves numerous processing decisions, and variations
in how each dataset represents the temperature field contribute to what is known as structural
uncertainty [Thorne]_. This structural uncertainty has two primary sources: (1) differences in the
processing of the input temperature measurements, and (2) differences in the spatial interpolation
methods applied. Because these steps are often tightly integrated, it is difficult to determine
their individual contributions.

`glomar_gridding` is a software package developed to support the evaluation of structural
uncertainty by offering flexible tools for spatial interpolation. The package enables users to
spatially interpolate grid-box average observations and their associated uncertainty estimates using
Gaussian Process Regression Modelling (GPRM, often called `Kriging`) [Rasmussen]_, [Cressie]_ . This
technique builds on established methods for generating surface temperature fields [Karspeck]_,
[Morice_2012]_. By decoupling interpolation from earlier processing stages - such as homogenization,
quality control, and aggregation to grid-cell averages - `glomar_gridding` allows users to create
spatially complete fields while independently assessing the effects of upstream data processing
choices.

========
Citation
========

Richard C. Cornes, Steven. C. Chan, Archie Cable et al. GloMarGridding: A Python Package for Spatial
Interpolation to Support Structural Uncertainty Assessment of Climate Datasets, 22 August 2025,
PREPRINT (Version 1) available at Research Square: https://doi.org/10.21203/rs.3.rs-7427869/v1.
