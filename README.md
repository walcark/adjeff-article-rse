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

The scripts for figure generation are stored in ``scripts/``, and generate figure stored in ``output/``. All the data required to perform computation are stored in ``data/`` (mainly Smart-G auxiliary data, but also some pre-treated data). 

The ``pyproject.toml`` is used by pixi (or any environment manager you may want to use) in order to install the dependencies required to compute the figures. If you wish to keep it simple, simple launch the ``install_env`` script that will locally install ``pixi`` on your system and automatically install the dependencies.

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

# Install pixi
chmod +x install_env
./install_env

# GPU environment (required for all figures)
pixi install -e gpu
```

### Download Smart-G auxiliary data

The scripts ``scripts/auxdata.py`` allows to download Smart-G auxiliary data with the same process described on the Smart-G GitHub page (https://github.com/hygeos/smartg). It can easily be launched with the following pixi command:

```bash
pixi run donwload_auxdata
```

### Compute a figure

The ``makefig`` shell file allows to simply call each ``scripts/figureX.py`` file, using the following command:

```bash
chmod +x makefig

./makefig figure2      # to compute the second figure
./makefig figureX      # to compute the Xth figure
./makefig              # to compute all the figures successively (quite long)
```

Output files are written to `article-scripts/figs/`.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
