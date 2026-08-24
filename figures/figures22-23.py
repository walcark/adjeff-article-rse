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
import scienceplots  # noqa: F401

plt.style.use(["science", "nature"])

DATA_DIR = Path(__file__).parent.parent / "data" / "scatterpoints_figures22-23"
FIGS_DIR = Path(__file__).parent.parent / "output"

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
BANDS_NM = [492, 560, 665, 705, 740, 783]
MARKERS = {
    "unif": "o",
    "maja": "s",
    "king": "^",
    "wu": "D",
}


def load_band(band: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"results_{band}_noglint.csv")
    res_unif = (df["pts_unif"] - df["pts_reel"]).abs()
    res_maja = (df["pts_maja"] - df["pts_reel"]).abs()
    res_king = (df["pts_king"] - df["pts_reel"]).abs()
    mean_res = (res_unif + res_maja + res_king) / 3
    threshold = mean_res.mean() + 3 * mean_res.std()
    return df[mean_res <= threshold].reset_index(drop=True)


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def pct_improvement(rmse_other: float, rmse_king: float) -> float:
    """Return RMSE improvement of King over another model (positive = King better)."""
    return (rmse_other - rmse_king) / rmse_other * 100


def main() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    tick_factor = 1.2

    # ── Figure 22: scatter plots ──────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for idx, (ax, band) in enumerate(zip(axes.flatten(), BANDS)):
        df = load_band(band)
        x = df["pts_reel"].values
        y_unif = df["pts_unif"].values
        y_maja = df["pts_maja"].values
        y_king = df["pts_king"].values
        y_wu = df["pts_wu"].values

        rmse_unif = rmse(y_unif, x)
        rmse_maja = rmse(y_maja, x)
        rmse_king = rmse(y_king, x)
        rmse_wu = rmse(y_wu, x)

        scatter_kw = dict(s=20, alpha=0.75, zorder=3)
        ax.set_title(f"({'abcdef'[idx]}) {band}", fontsize=12 * tick_factor, pad=10)
        ax.scatter(
            x,
            y_unif,
            marker=MARKERS["unif"],
            label=rf"$\mathbf{{Unif}}: {rmse_unif:.4f}$",
            **scatter_kw,
        )
        ax.scatter(
            x,
            y_maja,
            marker=MARKERS["maja"],
            label=rf"$\mathbf{{Gauss}}: {rmse_maja:.4f}$",
            **scatter_kw,
        )
        ax.scatter(
            x,
            y_wu,
            marker=MARKERS["wu"],
            label=rf"$\mathbf{{Wu}}: {rmse_wu:.4f}$",
            **scatter_kw,
        )
        ax.scatter(
            x,
            y_king,
            marker=MARKERS["king"],
            label=rf"$\mathbf{{King}}:~{rmse_king:.4f}$",
            **scatter_kw,
        )

        all_vals = np.concatenate([x, y_unif, y_maja, y_king, y_wu])
        nice_max = np.ceil(all_vals.max() / 0.05) * 0.05
        lim = (0.0, nice_max)
        ax.plot(lim, lim, color="black", linewidth=2, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")

        if idx >= 3:
            ax.set_xlabel(r"$\rho_\mathrm{s}$", fontsize=12 * tick_factor)
        if idx in [0, 3]:
            ax.set_ylabel(r"$\hat{\rho}_\mathrm{s}$", fontsize=12 * tick_factor)
        ax.tick_params(
            axis="both",
            which="major",
            width=1.5,
            length=6,
            labelsize=10 * tick_factor,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            width=1.0,
            length=3,
            labelsize=8 * tick_factor,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.legend(fontsize=10 * tick_factor, loc="lower right", framealpha=0.9)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / "figure22.png", dpi=600)
    plt.show()

    # ── Figure 23: RMSE improvement vs wavelength ─────────────────────────────
    impr_unif, impr_maja, impr_wu = [], [], []
    for band in BANDS:
        df = load_band(band)
        x = df["pts_reel"].values
        r_unif = rmse(df["pts_unif"].values, x)
        r_maja = rmse(df["pts_maja"].values, x)
        r_wu = rmse(df["pts_wu"].values, x)
        r_king = rmse(df["pts_king"].values, x)
        impr_unif.append(pct_improvement(r_unif, r_king))
        impr_maja.append(pct_improvement(r_maja, r_king))
        impr_wu.append(pct_improvement(r_wu, r_king))

    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    ax2.plot(
        BANDS_NM,
        impr_unif,
        marker=MARKERS["unif"],
        linewidth=1.5,
        markersize=7,
        label="vs Unif",
    )
    ax2.plot(
        BANDS_NM,
        impr_maja,
        marker=MARKERS["maja"],
        linewidth=1.5,
        markersize=7,
        label=r"vs Gauss",
    )
    ax2.plot(
        BANDS_NM,
        impr_wu,
        marker=MARKERS["wu"],
        linewidth=1.5,
        markersize=7,
        label="vs Wu",
    )
    ax2.set_ylim(-10, 70)
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_xlim(BANDS_NM[0], BANDS_NM[-1])
    ax2.set_xlabel(r"Wavelength $[\mathrm{nm}]$", fontsize=12 * tick_factor)
    ax2.set_ylabel(r"RMSE improvement $[\%]$", fontsize=12 * tick_factor)
    ax2.tick_params(
        axis="both",
        which="major",
        width=1.5,
        length=6,
        labelsize=10 * tick_factor,
    )
    ax2.tick_params(
        axis="both",
        which="minor",
        width=1.5,
        length=3,
        labelsize=8 * tick_factor,
    )
    ax2.tick_params(axis="both", pad=5)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    ax2.legend(fontsize=10 * tick_factor, framealpha=0.9)

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / "figure23.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
