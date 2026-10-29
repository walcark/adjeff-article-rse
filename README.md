# Adjeff - article companion scripts

This reposity contains most of the scripts used to generate the results from the article:

> K. Walcarius, M. Moulana, D. Ramon, M. Lassalle, J. Colin and O. Hagolle, *A Revised Model for Atmospheric Adjacency Effect Correction in Optical Satellite Earth Observation Images*, Remote Sensing of Environment, submitted.

Every figure of the article is recomputed here, from the Monte-Carlo radiative transfer up to the fitted point spread functions, through [adjeff](https://github.com/walcark/Adjeff) 1.2.7 and [Smart-G](https://github.com/hygeos/smartg) 2.0. The datasets deposited on the [SEDOO portal](https://www.sedoo.fr/donnees-surfaces-continentales/?uuid=a113b52e-7bad-4d41-8733-0f6679da6f1a) are built here too.

---
## Reproduce a figure

A CUDA GPU is required: every figure runs Smart-G. `install_env` installs the [pixi](https://pixi.prefix.dev/latest/) environment manager and the environment.

```bash
git clone https://github.com/walcark/adjeff-article-rse.git
cd adjeff-article-rse

chmod +x install_env makefig
./install_env                 # installs pixi into ~/.pixi, then the environment

./makefig figure7             # one figure
./makefig                     # all of them, successively
```

Figures are saved in `results/figures/`.

### Available scripts

| `./makefig` | Output | Script location |
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

> [!note] King kernel LUT computation
> The script `generate/sedoo/build_psf_lut.py` allows to compute the full look-up table of the king model parameters for all the CAMS aerosols. It requires very long computations but can be adapted > to other aerosols.

---
## Layout of the repository

```
generate/               everything that produces something
├── figures/            the article's figures
├── reviews/            analyses supporting the responses to the reviewers
├── sedoo/              the datasets deposited on the SEDOO portal
└── smoke.py            checks every script above still runs (for development only)

inputs/                 what the figures scripts require
├── maja/               the Cairo scene of figure 17, reduced to 9 MB
├── modis/              the MODIS skylight table and BRDF weights
├── gloria/             pre-processed GLORIA matchups
└── smartg_auxdata/     molecular cross sections, AFGL profiles, CAMS aerosols

results/                outputs, ignored by git
├── figures/         
├── sedoo/           
└── smoke/           

tools/                  neither produced nor required
└── clamp_brdf.py       turns Smart-G's Ross-Li clamp on and off

src/adjeff_article_1/   main utility scripts used in the repository
makefig                 builds one figure or all of them
install_env             installs pixi and the environment
```

Nothing is downloaded: 

`inputs/smartg_auxdata/` holds a minimal subset of the [Smart-G dataset](https://github.com/hygeos/smartg). Additional aerosol tables for the CAMS aerosols were generated with the MOPSMAP Fortran code through [pymopsmap](https://github.com/walcark/pymopsmap). The script `generate/sedoo/build_cams_auxdata.py` documents the grid that they are tabulated on.

The Smart-G cache sits outside the repository, under `~/.cache/adjeff-article/`: it runs to gigabytes and has to survive a reboot. [docs/MAINTAINING.md](docs/MAINTAINING.md) says how to move it, and covers the rest of what reproducing a figure does not need.

## License

Apache License 2.0, see [LICENSE](LICENSE).
