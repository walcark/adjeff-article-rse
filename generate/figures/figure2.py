"""Surface reflectance with a generalised Gaussian PSF (figure 2).

A generalised Gaussian is fitted on three disks of radius 1, 5 and 50 km,
then used to retrieve the surface reflectance of a scene.  Three curves
are compared: no adjacency correction at all, the fixed 330 m Gaussian
MAJA applies, and the fitted kernel.
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
        name="figure2",
        band=BAND,
        run=run,
        psf_type=GeneralizedGaussianPSF,
        init_parameters={"sigma": 1e-3, "n": 0.20},
        label="GG",
    )


if __name__ == "__main__":
    main()
