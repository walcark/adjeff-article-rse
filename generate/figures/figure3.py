"""Surface reflectance with a King PSF (figure 3).

Figure 2 with a King kernel in place of the generalised Gaussian: fitted
on the same three disks, compared against no correction and against the
fixed 330 m Gaussian.
"""

from adjeff.core import KingPSF, S2Band

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

    # Optimize the King model, and plot the comparison
    psf_comparison_figure(
        name="figure3",
        band=BAND,
        run=run,
        psf_type=KingPSF,
        init_parameters={"sigma": 0.1, "gamma": 2.0},
        label="King",
    )


if __name__ == "__main__":
    main()
