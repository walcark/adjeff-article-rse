# Maintaining this repository

- [Logging](#logging)
- [Moving the cache](#moving-the-cache)
- [Building the SEDOO lookup table](#building-the-sedoo-lookup-table)
- [Earthdata credentials, for `hotspot`](#earthdata-credentials-for-hotspot)
- [Where the BRDF coefficients come from](#where-the-brdf-coefficients-come-from)
- [Checking the scripts still run](#checking-the-scripts-still-run)

---

## Logging

adjeff prints nothing until asked, so the figure scripts ask on your behalf.
Every one of them takes:

```bash
./makefig figure2                                        # info, the default
python generate/figures/figure2.py --log-level debug     # + cache, atmospheres
python generate/figures/figure2.py --log-level warning   # only what went wrong
python generate/figures/figure2.py --log-json            # one JSON object a line
```

---

## Moving the cache

The Smart-G cache is under `~/.cache/adjeff-article/`, outside the repository
and not in `/tmp`: it runs to gigabytes and has to survive a reboot.

| | |
|---|---|
| `ADJEFF_ARTICLE_CACHE` | moves it wholesale, which is what a cluster needs: the quota is on `$HOME`, the room is on the scratch |
| `--cache-dir` | moves it for one run alone, on every script |

---

## Building the SEDOO lookup table

The tables are already on the
[SEDOO portal](https://www.sedoo.fr/donnees-surfaces-continentales/?uuid=a113b52e-7bad-4d41-8733-0f6679da6f1a).
This is how they were produced, for whoever needs another grid.

`generate/sedoo/build_psf_lut.py` fits the King PSF over 12 bands, 9 aerosol
optical thicknesses, 4 humidities, 3 reference heights and 2 ground elevations:
2592 optimisations per species, eight species. It therefore runs on a cluster,
one species per job:

```bash
mkdir -p sbatch_reports     # once: sbatch resolves --output at submit time
sbatch generate/sedoo/build_psf_lut.slurm 1 1          # one slab, to time it
for g in $(seq 8); do sbatch generate/sedoo/build_psf_lut.slurm $g; done
```

A slab is one `(band, href)`, its 72 states swept in one go, and takes about
1h40 on a V100. Thirty-six slabs make 59 hours, which is why a job carries one
species rather than two.

Each slab is written under `slabs/` as soon as it is fitted, so a job that runs
out of time is resubmitted and skips what is there. That needs the cache and the
slabs to outlive the job, hence both on the scratch.

The cache of a slab is deleted once its file is written, which is what keeps the
scratch from filling: a state weighs 128 MB at 3999 pixels, a slab some 55 GB,
a species 2 TB. Nothing downstream reads it, the next slab rebuilding its config
around another band, reference height or species. `--keep-cache` turns the
deletion off, for debugging a slab whose scenes are worth keeping.

`--training gauss` is what the deposited tables were fitted on; `--training
disks` is what the figures of the article use. A cube built with the other one
is not a reproduction of the deposit.

---

## Earthdata credentials, for `hotspot`

The MODIS BRDF products the hotspot study uses are served from hosts that ask
who you are. Nothing of the sort belongs in this repository, so the credentials
are read from outside it, in this order:

1. `--earthdata-user` and `--earthdata-password`, or `--earthdata-token`;
2. `EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`;
3. a TOML file, `~/.config/adjeff-article/credentials.toml` by default,
   moved with `--credentials` or `$ADJEFF_ARTICLE_CREDENTIALS`;
4. `~/.netrc`, which is what NASA's own documentation asks you to write.

```bash
mkdir -p ~/.config/adjeff-article
$EDITOR ~/.config/adjeff-article/credentials.toml
chmod 600 ~/.config/adjeff-article/credentials.toml
```

```toml
[earthdata]
username = "your-login"
password = "your-password"
# or, instead of both, a token you can revoke without changing the account:
# token = "…"
```

That last `chmod` is checked rather than assumed: a file created with the
default umask is world-readable, and the loader says so. It logs which source it
used and the username, never the secret.

---

## Where the BRDF coefficients come from

Two services carry MCD43A1, and `hotspot` takes `--brdf-source`:

| | account | speed | state |
| --- | --- | --- | --- |
| `ornl` | none | seconds | catalogue currently down |
| `appeears` | Earthdata | ~6 min per fetch | working |
| `auto` *(default)* | | | tries ORNL, falls back to AppEEARS |

```bash
python generate/reviews/hotspot.py                        # ORNL, then AppEEARS
python generate/reviews/hotspot.py --brdf-source appeears # skip ORNL
python generate/reviews/hotspot.py --refresh-sites        # ignore the CSV
```

The ORNL TESViS service answers `500` to every request touching its catalogue,
while `/networks`, whose answer is a static list, succeeds. The failure is in
their database layer and no account gets past it: an outage, not a refusal.

AppEEARS is LP DAAC's point-sampling service for the same granules. It submits a
task and collects it once it has run, so a fetch takes minutes rather than
seconds. It happens once: the result is cached to `--sites-csv`, and later runs
read the CSV without touching either service. Any CSV carrying `site`, `f_iso`,
`f_geo` and `f_vol` is accepted, so an Earth Engine export feeds the study just
as well.

---

## Checking the scripts still run

After bumping the pinned adjeff release, run every script at a tiny grid and
photon count. It proves the API each one calls still exists and still returns
usable shapes, and nothing about the physics:

```bash
pixi run -e default python generate/smoke.py --cold
```

`--cold` is what a version check needs: a warm cache short-circuits the Smart-G
calls, so a run that reuses it only proves that zarr still reads back. A script
joins the run automatically as soon as its `--help` advertises `--smoke`.
