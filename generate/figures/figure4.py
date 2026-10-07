"""Comparison of rho_s predictions: GeneralizedGaussianPSF vs Gauss-330.

Optimize a Generalized Gaussian PSF model of three disks of increasing radii
(1, 5, 50 km), then compares to surface reflectance estimate of the PSF model
with:

    - no adjacency effects correction (rho_unif),
    - adjacency effects correction with a Gaussian Kernel (sigma=330m)
"""

from adjeff.core import GeneralizedGaussianPSF, S2Band

from adjeff_article_1.psf_comparison import psf_comparison_figure
from adjeff_article_1.runconfig import parse_run

# Global parameters
BAND = S2Band.B03
RES_KM = 0.05
N = 3999


def main() -> None:

    # Parse input parameters
    run, _ = parse_run(__doc__.splitlines()[0])
    run = run.resolve(n=N, res_km=RES_KM)

    # Optimize the GG model, and plot the comparison
    psf_comparison_figure(
        name="figure4",
        band=BAND,
        run=run,
        psf_type=GeneralizedGaussianPSF,
        init_parameters={"sigma": 1e-3, "n": 0.20},
        label="GG",
    )


if __name__ == "__main__":
    main()
