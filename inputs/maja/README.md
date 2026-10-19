# Reduced Cairo scene

What `generate/figures/figure17.py` reads, and nothing else: a MAJA L2A
product cut down from 2.8 GB to 8 MB, so that the transect of Figure 17
runs from a clone.

| | |
|---|---|
| `..._SRE_B8A.tif` | the only band the figure uses, at 120 m |
| `..._ATB_R2.tif` | the aerosol optical thickness, at 120 m |
| `DEM/.../..._ALT_R2.TIF` | the ground elevation, at 120 m |
| `..._FRE_B2.tif` | 1200 m; only its bounds, transform and CRS are read |
| `..._MTD_ALL.xml` | the viewing and solar angles, and the humidity |
| `MASKS/*_R1.tif` | untouched; sensorsio opens all of them |

The three rasters that carry data were averaged by blocks of 6, from the
20 m of the original to the 120 m the figure works at, which is the
reduction the loader used to perform itself at every run. The figure is
therefore the same as the one the full product gives.

`..._FRE_B2.tif` holds the blue band of the SRE product rather than an FRE
one, which this product does not ship: sensorsio's `Sentinel2` constructor
opens a FRE B2 whatever band is asked for, and reads nothing but its grid.

`generate/sedoo/shrink_maja.py` built it. The full product is tile T36RVV
of 2023-06-02, on <https://catalogue.theia-land.fr>.
