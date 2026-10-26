# adjeff, companion scripts

Companion repository for:

> K. Walcarius, M. Moulana, D. Ramon, M. Lassalle, J. Colin and O. Hagolle, *A Revised Model for Atmospheric Adjacency Effect Correction in Optical Satellite Earth Observation Images*, Remote Sensing of Environment, submitted.

Every figure of the article is recomputed here, from the Monte-Carlo radiative transfer up to the fitted point spread functions, through [adjeff](https://github.com/walcark/Adjeff) 1.2.7 and [Smart-G](https://github.com/hygeos/smartg) 2.0. The datasets deposited on the [SEDOO portal](https://www.sedoo.fr/donnees-surfaces-continentales/?uuid=a113b52e-7bad-4d41-8733-0f6679da6f1a) are built here too.

---

## Reproduce a figure

A CUDA GPU is required: every figure runs Smart-G. `install_env` brings its own pixi and installs the environment, so nothing else is needed.

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

A target is the number the article prints: `figure5` builds Fig. 5.

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
| `figureC1`, `figureC2` | the two figures of Appendix C | `figures/figureC1.py`, `figureC2.py` |
| `table_aot_sensitivity` | what a 0.2 error on the AOT costs, per band | `reviews/table_aot_sensitivity.py` |

One more target is not a figure of the article: `elevation_low_aot` supports section 3.2.2 on the elevation dependency below AOT 0.1. The header of `makefig` lists the figures built outside this repository.

---

## How it is laid out

```
generate/            everything that produces something
├── figures/         the article's figures
├── reviews/         analyses supporting the responses to the reviewers
├── sedoo/           the datasets deposited on the SEDOO portal
└── smoke.py         checks every script above still runs

inputs/              what the figures read and this repository does not produce
├── maja/            the Cairo scene of figure 17, reduced to 9 MB
├── modis/           the MODIS skylight table and BRDF weights
├── gloria/          pre-processed GLORIA matchups
└── smartg_auxdata/  molecular cross sections, AFGL profiles, CAMS aerosols

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

Nothing is downloaded: `inputs/smartg_auxdata/` holds the subset of the [Smart-G dataset](https://github.com/hygeos/smartg) these figures touch, at the paths Smart-G reads them from, 6 MB beside the 168 MB of aerosol tables.

Those aerosol tables came from MOPSMAP through [pymopsmap](https://github.com/walcark/pymopsmap); `generate/sedoo/build_cams_auxdata.py` documents the grid they are tabulated on, and rebuilding them needs the MOPSMAP binary, which reproducing the figures does not.

`pyproject.toml` is what pixi reads, and any other environment manager can read it too.

The Smart-G cache sits outside the repository, under `~/.cache/adjeff-article/`: it runs to gigabytes and has to survive a reboot. [docs/MAINTAINING.md](docs/MAINTAINING.md) says how to move it, and covers the rest of what reproducing a figure does not need.

## License

Apache License 2.0, see [LICENSE](LICENSE).
