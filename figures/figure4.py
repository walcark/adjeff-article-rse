"""Comparison of rho_s predictions: GeneralizedGaussianPSF vs Gauss-330.

Three disk training fields (radii 1, 5, 50 km) are run through the
forward pipeline.  A GeneralizedGaussianPSF is then optimised with
RMSE_RAD loss, and its predictions are compared to a fixed Gauss-330m
reference.
"""

from adjeff.core import GeneralizedGaussianPSF, S2Band
from adjeff_article_1.psf_comparison import psf_comparison_figure
from adjeff_article_1.runconfig import parse_run

BAND = S2Band.B03


def main() -> None:
    run, _ = parse_run(__doc__.splitlines()[0])
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
