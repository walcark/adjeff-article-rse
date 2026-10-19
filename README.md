# adjeff, companion scripts

Companion repository for:

> K. Walcarius, M. Moulana, D. Ramon, M. Lassalle, J. Colin and O. Hagolle,
> *A Revised Model for Atmospheric Adjacency Effect Correction in Optical
> Satellite Earth Observation Images*, Remote Sensing of Environment,
> submitted.

Every figure of the article is recomputed here, from the Monte-Carlo
radiative transfer up to the fitted point spread functions, through
[adjeff](https://github.com/walcark/Adjeff) 1.2.5 and
[Smart-G](https://github.com/hygeos/smartg) 2.0. The datasets deposited on
the SEDOO portal are built here too.

---

## Reproduce a figure

A CUDA GPU is required: every figure runs Smart-G. `install_env` brings
its own pixi and installs the environment, so nothing else is needed.

```bash
git clone https://github.com/walcark/adjeff-article-rse.git
cd adjeff-article-rse

chmod +x install_env makefig
./install_env                 # installs pixi into ~/.pixi, then the environment

./makefig figure7             # one figure
./makefig                     # all of them, successively, which is long
```

Figures land in `results/figures/`, tables in `results/tables/`.

### What each target draws

| `./makefig` | What it shows | Script |
|---|---|---|
| `figure2` | surface reflectance, generalised Gaussian PSF against Gauss-330 | `figures/figure2.py` |
| `figure3` | the same, with the King PSF | `figures/figure3.py` |
| `figure5` to `figure8` | PSF against aerosol optical thickness: black carbon, sulphate, sulphate at 95 % humidity, sea salt | `figures/figure5_15.py` |
| `figure9` to `figure11` | PSF against wavelength, one species each | `figures/figure5_15.py` |
| `figure12` | PSF against the aerosol scale height | `figures/figure5_15.py` |
| `figure13` | PSF against the ground elevation | `figures/figure5_15.py` |
| `figure14` | PSF against the viewing zenith angle | `figures/figure5_15.py` |
| `figure15` | PSF against the solar zenith angle | `figures/figure5_15.py` |
| `figure17` | reflectance transect across the Cairo tile, with its TOA estimate | `figures/figure17.py` |
| `figures20-21` | King PSF on real Sentinel-2 data, scatter and RMSE gain | `figures/figures20-21.py` |
| `figureC1` | loss landscape of the generalised Gaussian PSF, four training sets | `figures/figureC1.py` |
| `figureC2` | encircled-energy radii against the radial RMSE loss | `figures/figureC2.py` |
| `table_aot_sensitivity` | what a 0.2 error on the AOT costs, per band, answering a reviewer | `reviews/table_aot_sensitivity.py` |

**A target is the number the article prints**: `figure5` builds Fig. 5. The
header of `makefig` lists what is missing from the table, the figures built
outside this repository among them.

`figureC1` and `figureC2` are the two figures of Appendix C, hence their
names. One target is not a figure of the article at all: `elevation_low_aot`
supports the statement of section 3.2.2 that the elevation dependency grows
as the AOT drops below 0.1.

---

## What the repository needs

Nothing is downloaded: `inputs/` holds what the figures read, at the paths
Smart-G reads them from.

| | |
|---|---|
| `inputs/smartg_auxdata/acs/` | molecular absorption cross sections |
| `inputs/smartg_auxdata/atmospheres/` | the AFGL profiles, exponential one included |
| `inputs/smartg_auxdata/aerosols/OPAC/mixtures/` | optical properties of the eight CAMS aerosol species |
| `inputs/maja/` | the Cairo scene of figure 17, cut down to what it reads |
| `inputs/modis/` | the MODIS skylight table and BRDF weights |
| `inputs/gloria/` | pre-processed GLORIA matchups |

That is the subset of the Smart-G auxiliary dataset these figures touch,
6 MB beside the 168 MB of aerosol tables. The rest of
[the full dataset](https://github.com/hygeos/smartg) is not needed.

The CAMS tables were produced with MOPSMAP, through
[pymopsmap](https://github.com/walcark/pymopsmap).
`generate/sedoo/build_cams_auxdata.py` produced them and documents the grid
they are tabulated on; running it needs the MOPSMAP binary and its optical
dataset, which reproducing the figures does not.

The Smart-G cache sits outside the repository, under
`~/.cache/adjeff-article/`: it runs to gigabytes, it is reproducible, and a
lookup table takes long enough that it has to survive a reboot.

---

## How it is laid out

Three directories carry the layout, and the name of each says what is in
it: `generate/` produces, `inputs/` is needed, `results/` is produced.

```
generate/            everything that produces something
├── figures/         the article's figures
├── reviews/         analyses supporting the responses to the reviewers
├── sedoo/           the datasets deposited on the SEDOO portal
└── smoke.py         checks every script above still runs

inputs/              what the repository needs and does not produce
├── maja/            the Cairo scene of figure 17, reduced to 8 MB
├── modis/           the MODIS skylight table and BRDF weights
├── gloria/          pre-processed GLORIA matchups
└── smartg_auxdata/  the subset of the Smart-G dataset the figures read

results/             everything produced, ignored by git
├── figures/         .png
├── tables/          .csv and .tex
├── sedoo/           what gets deposited
└── smoke/           the smoke run's own output

tools/               neither produced nor required
└── clamp_brdf.py    turns Smart-G's Ross-Li clamp on and off

src/adjeff_article_1/   the library every script under generate/ imports
makefig                 builds one figure or all of them
install_env             installs pixi and the environment
```

`pyproject.toml` is what pixi reads, and any other environment manager can
read it too. `pixi install -e local` builds adjeff from a sibling checkout
of `../Adjeff` rather than from PyPI, which is how a change is tried before
it is released.

---

## Going further

[docs/MAINTAINING.md](docs/MAINTAINING.md) covers what reproducing a figure
does not need: the log levels, the Earthdata credentials the hotspot study
asks for, where the MODIS BRDF coefficients come from, moving the cache,
and building the SEDOO lookup table on a cluster.

## License

Apache License 2.0, see [LICENSE](LICENSE).
