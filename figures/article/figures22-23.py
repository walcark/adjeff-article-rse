"""Scatter plots and RMSE improvement for King PSF on Sentinel-2 bands.

Two figures built from real Sentinel-2 data:

  - Figure 22: scatter plots of predicted vs. actual rho_s for six bands,
               comparing Unif, Gauss (MAJA), Wu, and King PSF corrections.
  - Figure 23: RMSE improvement of King vs. Unif/Gauss/Wu as a function of
               wavelength.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adjeff_article_1.runconfig import REPO_ROOT, parse_run
from adjeff_article_1.style import (
    font,
    panel_title,
    save,
    style_axes,
    use_article_style,
)

DATA_DIR = REPO_ROOT / "data" / "scatterpoints_figures22-23"

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
BANDS_NM = [492, 560, 665, 705, 740, 783]

# Corrections compared against the reference reflectance, in plotting
# order: (column, marker, legend name).  King is the model of this article
# and is drawn last so that it sits above the others.
MODELS = [
    ("pts_unif", "o", "Unif"),
    ("pts_maja", "s", "Gauss"),
    ("pts_wu", "D", "Wu"),
    ("pts_king", "^", "King"),
]
REFERENCE = "pts_reel"

# Columns entering the outlier rejection.  Wu is deliberately left out so
# that the retained points do not depend on it.
OUTLIER_COLUMNS = ["pts_unif", "pts_maja", "pts_king"]


def load_band(band: str) -> pd.DataFrame:
    """Return the scatter points of *band*, three sigma outliers removed."""
    df = pd.read_csv(DATA_DIR / f"results_{band}_noglint.csv")
    residuals = [
        (df[column] - df[REFERENCE]).abs() for column in OUTLIER_COLUMNS
    ]
    mean_res = sum(residuals) / len(residuals)
    threshold = mean_res.mean() + 3 * mean_res.std()
    return df[mean_res <= threshold].reset_index(drop=True)


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Return the root mean squared error between *pred* and *ref*."""
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def pct_improvement(rmse_other: float, rmse_king: float) -> float:
    """Return the RMSE gain of King over another model, in percent."""
    return (rmse_other - rmse_king) / rmse_other * 100


def band_rmse(df: pd.DataFrame) -> dict[str, float]:
    """Return ``{column: rmse}`` for every model of *df*."""
    reference = df[REFERENCE].values
    return {
        column: rmse(df[column].values, reference) for column, _, _ in MODELS
    }


def scatter_figure(figs_dir: Path) -> None:
    """Draw figure 22: predicted vs actual reflectance, one panel per band."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for idx, (ax, band) in enumerate(zip(axes.flatten(), BANDS)):
        df = load_band(band)
        reference = df[REFERENCE].values
        errors = band_rmse(df)

        for column, marker, name in MODELS:
            ax.scatter(
                reference,
                df[column].values,
                marker=marker,
                label=rf"$\mathbf{{{name}}}: {errors[column]:.4f}$",
                s=20,
                alpha=0.75,
                zorder=3,
            )

        columns = [reference] + [df[c].values for c, _, _ in MODELS]
        nice_max = np.ceil(np.concatenate(columns).max() / 0.05) * 0.05
        lim = (0.0, nice_max)
        ax.plot(lim, lim, color="black", linewidth=2, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")

        panel_title(ax, idx, band, pad=10)
        if idx >= 3:
            ax.set_xlabel(r"$\rho_\mathrm{s}$", fontsize=font())
        if idx in (0, 3):
            ax.set_ylabel(r"$\hat{\rho}_\mathrm{s}$", fontsize=font())
        style_axes(ax)
        ax.legend(fontsize=font(10 / 12), loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save(fig, "figure22", figs_dir)


def improvement_figure(figs_dir: Path) -> None:
    """Draw figure 23: King's RMSE gain over each model, versus wavelength."""
    errors = [band_rmse(load_band(band)) for band in BANDS]

    fig, ax = plt.subplots(figsize=(5, 3.6))

    for column, marker, name in MODELS:
        if column == "pts_king":
            continue
        gains = [pct_improvement(e[column], e["pts_king"]) for e in errors]
        ax.plot(
            BANDS_NM,
            gains,
            marker=marker,
            linewidth=1.5,
            markersize=7,
            label=f"vs {name}",
        )

    ax.set_ylim(-10, 70)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(BANDS_NM[0], BANDS_NM[-1])
    ax.set_xlabel(r"Wavelength $[\mathrm{nm}]$", fontsize=font())
    ax.set_ylabel(r"RMSE improvement $[\%]$", fontsize=font())
    style_axes(ax)
    ax.tick_params(axis="both", pad=5)
    ax.legend(fontsize=font(10 / 12), framealpha=0.9)

    fig.tight_layout()
    save(fig, "figure23", figs_dir)


def main() -> None:
    run, _ = parse_run(__doc__.splitlines()[0])
    use_article_style()

    scatter_figure(run.figs_dir)
    improvement_figure(run.figs_dir)

    if not run.smoke:
        plt.show()


if __name__ == "__main__":
    main()
