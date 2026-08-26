# Adjeff — Remote Sensing of Environment (RSE) companion scripts

Companion repository for **Article 1**: *[title to be added]*.

Contains the scripts that reproduce every figure in the paper, and the associated datasets. 

---

## Repository structure and guidelines

The structure of the repository is organized as follow.

```
# Main folders of interest
├── figures/
│   ├── figure2.py                      # Script to generate figure 2
│   ├── figure3.py                      # Script to generate figure 3
│   ├── ...
├── output/                             # Contains output figures
├── data/                               # Contains data useful for figure generation
    ├── afgl_auxdata/
    ├── cams_aer_auxdata/
├── pyproject.toml                      # Useful to manage pixi environments
├── install_env                         # Simple script to install pixi (if necessary) and the project environment 
├── makefig                             # Main script to compute figures with adjeff
├── README.md                           # Instructions on how to use the repository


# Not important to the user
├── scripts/                            # Contains useful help scripts
    ├── auxdata.py                      # Download Smart-G auxdata and add additionnal useful data
    ├── export_smartg_auxdata.py        # Useful to export the Smart-G env variable before computation
├── src/                                # Source modules used in the scripts
└── LICENSE                             # Apache 2.0

```

The scripts for figure generation are stored in ``figures/``, and generate figures stored in ``output/``. All the data required to perform computation are stored in ``data/`` (mainly Smart-G auxiliary data, but also some pre-treated data). 

The ``pyproject.toml`` is used by pixi (or any environment manager you may want to use) in order to install the dependencies required to compute the figures. If you wish to keep it simple, simply launch the ``install_env`` script that will locally install ``pixi`` on your system and automatically install the dependencies.

The ``makefig`` helps to easily compute any figure on the terminal. Keep in mind that depending on your GPU capacities, some figures may take some time to fully compute.

---

## Installation

### Prerequisites

- [pixi](https://pixi.sh) ≥ 0.40
- A CUDA 12.6-compatible GPU and driver

### Clone the project, add pixi and launch an environment

```bash
# Clone the repository
git clone https://github.com/walcark/adjeff-article-1.git
cd adjeff-article-1

# Install pixi (to ~/.pixi by default)
chmod +x install_env
./install_env

# Or specify a custom installation directory
./install_env /custom/path

# GPU environment (required for all figures)
pixi install -e gpu
```

### Download Smart-G auxiliary data

The scripts ``scripts/auxdata.py`` allows to download Smart-G auxiliary data with the same process described on the Smart-G GitHub page (https://github.com/hygeos/smartg). It can easily be launched with the following pixi command:

```bash
pixi run download_auxdata
```

### Compute a figure

The ``makefig`` shell file allows to simply call each ``figures/figureX.py`` file, using the following command:

```bash
chmod +x makefig

./makefig figure2      # to compute the second figure
./makefig figureX      # to compute the Xth figure
./makefig              # to compute all the figures successively (quite long)
```

Output files are written to `output/`.

### Logging

adjeff 0.13.0 prints nothing until asked, so the figure scripts ask on
your behalf. Every one of them takes:

```bash
./makefig figure4                                   # info, the default
python figures/figure4.py --log-level debug         # + cache, atmospheres
python figures/figure4.py --log-level warning       # only what went sideways
python figures/figure4.py --log-json                # one JSON object per line
```

At `info` you get, for each module, what it is about to run and what it
cost: `sweep.plan` before the Smart-G call and `module.done` with its
duration after, plus `xsweep`'s own point counts and elapsed times. `zarr`
and the other libraries are capped at `warning`.

### Earthdata credentials, for `hotspot`

The MODIS BRDF products the hotspot study uses are served from hosts that
ask who you are. Nothing of the sort belongs in this repository, so the
credentials are read from outside it, in this order:

1. `--earthdata-user` and `--earthdata-password`, or `--earthdata-token`;
2. `EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`;
3. a TOML file, `~/.config/adjeff-article/credentials.toml` by default,
   moved with `--credentials` or `$ADJEFF_ARTICLE_CREDENTIALS`;
4. `~/.netrc`, which is what NASA's own documentation asks you to write.

The file:

```toml
[earthdata]
username = "your-login"
password = "your-password"
# or, instead of both, a token you can revoke without changing the account:
# token = "…"
```

```bash
mkdir -p ~/.config/adjeff-article
$EDITOR ~/.config/adjeff-article/credentials.toml
chmod 600 ~/.config/adjeff-article/credentials.toml
```

The last step is checked rather than assumed: a file created with the
default umask is world-readable, and the loader says so. It logs which
source it used and the username, never the secret.

#### Where the BRDF coefficients come from

Two services carry MCD43A1, and `hotspot` takes `--brdf-source`:

| | account | speed | state |
| --- | --- | --- | --- |
| `ornl` | none | seconds | catalogue currently down |
| `appeears` | Earthdata | ~6 min per fetch | working |
| `auto` *(default)* | — | — | tries ORNL, falls back to AppEEARS |

The ORNL TESViS service answers `500` to every request touching its
catalogue: `/products`, `/sites` and `/{product}/bands` all fail, while
`/networks`, whose answer is a static list, succeeds. The failure is in
their database layer, and **no account gets past it** — a 500 is an
outage, not a refusal, and the error message says so.

AppEEARS is LP DAAC's point-sampling service for the same granules. It
submits a task and collects it once it has run, so a fetch takes minutes
rather than seconds, and it only happens once: the result is cached to
`--sites-csv`. Later runs read the CSV and never touch either service.

```bash
python figures/hotspot.py                          # ORNL, then AppEEARS
python figures/hotspot.py --brdf-source appeears   # skip the ORNL attempt
python figures/hotspot.py --refresh-sites          # refetch, ignoring the CSV
```

Any CSV carrying `site`, `f_iso`, `f_geo` and `f_vol` is accepted, so an
Earth Engine export or a local granule read feeds the study just as well.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
