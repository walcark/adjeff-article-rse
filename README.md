# adjeff — Article 1 companion scripts

Companion repository for **Article 1**: *[title to be added]*.

Contains the scripts that reproduce every figure in the paper, and the
associated datasets.

---

## Repository structure

```
adjeff-article-1/
├── article-scripts/
│   ├── figure2.py          # Loss landscape — GaussGeneralPSF
│   ├── figure3.py          # Scatter: RMSE loss vs encircled-energy radius
│   ├── figure4.py          # rho_s predictions: GaussGeneralPSF vs Gauss-330
│   ├── figure5.py          # rho_s predictions: KingPSF vs Gauss-330
│   ├── figure7_17.py       # PSF sensitivity (aot, rh, wl, h, href, vza)
│   ├── generate_article_figures.sh
│   └── figs/               # output figures (created at runtime)
├── data/
│   └── ...                 # see Data section below
├── LICENSE                 # Apache 2.0
├── pyproject.toml
└── README.md
```

---

## Data

<!-- Describe the content of the data/ directory here.
     Replace the example tree below with the actual structure. -->

```
data/
└── (example — replace with your actual layout)
```

---

## Installation

### Prerequisites

- [pixi](https://pixi.sh) ≥ 0.40
- A CUDA 12.6-compatible GPU and driver
- Smart-G auxiliary data — set `SMARTG_DIR_AUXDATA` before running (see below)

### Install

```bash
git clone https://github.com/walcark/adjeff-article-1.git
cd adjeff-article-1

# GPU environment (required for all figures)
pixi install -e gpu
```

To use a local development version of adjeff instead of the PyPI release,
edit `pyproject.toml` and uncomment the local path dependency:

```toml
# adjeff = { path = "../adjeff", editable = true }
```

### Smart-G auxiliary data

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
```

---

## Generating the figures

### All figures at once

```bash
SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata \
    pixi run -e gpu figures-all
```

Or directly:

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
bash article-scripts/generate_article_figures.sh
```

### A single figure

```bash
export SMARTG_DIR_AUXDATA=/path/to/smartg/auxdata
bash article-scripts/generate_article_figures.sh figure7
```

Available figures: `figure2`, `figure3`, `figure4`, `figure5`,
`figure7` … `figure17`.

Output files are written to `article-scripts/figs/`.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
