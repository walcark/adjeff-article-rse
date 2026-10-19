"""Cut a MAJA product down to what the Cairo transect reads.

Writes `inputs/maja/`, 9 MB against the 2.8 GB of the full product and
its DEM, so that figure 17 runs from a clone.  Every reduction matches
what the loader computes at each run, so the figure does not change;
`inputs/maja/README.md` says which file holds what.

    python generate/sedoo/shrink_maja.py
"""
import shutil
from pathlib import Path

import numpy as np
import rasterio
from adjeff.modules.loaders.maja_loader import downsample_res
from rasterio.enums import Resampling

SRC = Path("/home/walcark/downloads/SENTINEL2A_20230602-084138-448_L2A_T36RVV_C_V1-0")
DEM = Path("/home/walcark/downloads/DEM/S2__TEST_AUX_REFDE2_36RVV_1001.DBL.DIR")
OUT = Path("inputs/maja")
FACTOR = 6  # 20 m -> 120 m


def shrink(src, dst, how, factor=FACTOR, as_float=False, floor=False):
    with rasterio.open(src) as s:
        h, w = s.height // factor, s.width // factor
        if as_float:
            # In numpy, and with the loader's own reduction: GDAL rounds
            # its average to the source dtype before anything sees it.
            bands = []
            for i in range(s.count):
                arr = s.read(i + 1).astype(float)
                if floor:  # the loader floors the DEM before reducing it
                    arr = np.where(arr < 0.0, 0.0, arr)
                bands.append(downsample_res(arr, 0.12, 0.02))
            data = np.stack(bands).astype(np.float32)
        else:
            data = s.read(out_shape=(s.count, h, w), resampling=how)
        profile = s.profile | {
            "height": h, "width": w,
            "transform": s.transform * s.transform.scale(factor, factor),
            "compress": "deflate", "predictor": 2,
            "tiled": True, "blockxsize": 256, "blockysize": 256,
        }
        if as_float:
            profile |= {"dtype": "float32", "predictor": 3}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as d:
        d.write(data)
    print(f"{dst.name:52s} {w}x{h}  {dst.stat().st_size / 1048576:6.2f} MB")


stem = SRC.name
product = OUT / stem
# Cubic, which is what sensorsio resamples a band with.
shrink(SRC / f"{stem}_SRE_B8A.tif", product / f"{stem}_SRE_B8A.tif", Resampling.cubic)
# Block means, which is what the loader computes itself.
shrink(SRC / f"{stem}_ATB_R2.tif", product / f"{stem}_ATB_R2.tif",
       Resampling.average, as_float=True)
shrink(DEM / "S2__TEST_AUX_REFDE2_36RVV_1001_ALT_R2.TIF",
       OUT / "DEM" / DEM.name / "S2__TEST_AUX_REFDE2_36RVV_1001_ALT_R2.TIF",
       Resampling.average, as_float=True, floor=True)
# Only its bounds, transform and CRS are ever read, so it goes coarse.
shrink(SRC / f"{stem}_SRE_B2.tif", product / f"{stem}_FRE_B2.tif",
       Resampling.average, factor=60)
shutil.copy2(SRC / f"{stem}_MTD_ALL.xml", product / f"{stem}_MTD_ALL.xml")
