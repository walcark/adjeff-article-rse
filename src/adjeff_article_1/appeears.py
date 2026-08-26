"""MCD43A1 BRDF coefficients from AppEEARS, when ORNL cannot serve them.

The hotspot study reads its RTLS weights from MCD43A1.  It used to get
them from the ORNL TESViS web service, which is convenient because it
needs no account: one GET per site and the coefficients come back.

That service is, at the time of writing, answering ``500`` to every
request that touches its catalogue: ``/products``, ``/sites`` and
``/{product}/bands`` all fail, while ``/networks``, whose answer is a
static list, succeeds.  The failure is in their database layer and no
request shaped differently gets past it.

AppEEARS is LP DAAC's point-sampling service for the same granules.  It
asks for a NASA Earthdata account, which is what
:mod:`adjeff_article_1.credentials` exists to supply, and it works by
submitting a task and collecting it once it has run, rather than by
answering in-line.  That is slower, minutes rather than seconds, and it
only has to happen once: the result is written to the CSV the study
already treats as its hand-off point.

Notes
-----
`BRDF_Albedo_Parameters_Band1` is a 3-D layer holding, in order, the
isotropic, volumetric and geometric weights, scaled by ``1e-3`` with
``32767`` for fill.  AppEEARS names the three components of a 3-D layer
by appending ``_0``, ``_1`` and ``_2``, which is checked here rather
than assumed: a renaming upstream should fail loudly and not quietly
produce a physically wrong triplet.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import datetime, timedelta

from ._logging import get_logger
from .credentials import Credentials

__all__ = ["AppeearsError", "fetch_mcd43a1"]

logger = get_logger(__name__)

#: Root of the AppEEARS API.
ROOT = "https://appeears.earthdatacloud.nasa.gov/api"

#: Product and the layer holding the three RTLS weights.
PRODUCT = "MCD43A1.061"

#: Suffixes AppEEARS gives the components of a 3-D layer, in the order
#: MCD43A1 stores them.
COMPONENTS = ("iso", "vol", "geo")

#: Fill value of the parameter layer, and the scale to apply.
FILL = 32767
SCALE = 1e-3


class AppeearsError(RuntimeError):
    """Raised when AppEEARS cannot be reached, refuses, or answers oddly."""


def fetch_mcd43a1(
    credentials: Credentials,
    sites: Sequence[tuple[str, float, float]],
    modis_band: int,
    start_date: str,
    end_date: str,
    *,
    timeout_s: float = 3600.0,
    poll_s: float = 20.0,
) -> list[dict[str, float | str]]:
    """Return one RTLS triplet per site, averaged over the period.

    Parameters
    ----------
    credentials : Credentials
        Earthdata login.  Must be ``available``.
    sites : sequence of (str, float, float)
        Site name, latitude and longitude.
    modis_band : int
        MODIS band number, 1 to 7.
    start_date, end_date : str
        Composite dates, ``AYYYYDDD``, as the ORNL path uses.
    timeout_s : float, optional
        How long to wait for the task before giving up (default 1 h).
    poll_s : float, optional
        Seconds between status checks (default 20).

    Returns
    -------
    list[dict]
        One row per site, with ``site``, ``lat``, ``lon``, ``f_iso``,
        ``f_vol``, ``f_geo`` and ``n_dates``.

    Raises
    ------
    AppeearsError
        If the credentials are missing or refused, the task fails, the
        wait times out, or the bundle is not shaped as expected.
    """
    if not credentials.available:
        raise AppeearsError(
            "AppEEARS needs a NASA Earthdata account. Put one in "
            "~/.config/adjeff-article/credentials.toml, in EARTHDATA_USERNAME "
            "and EARTHDATA_PASSWORD, or in ~/.netrc; see "
            "adjeff_article_1.credentials."
        )

    layer = f"BRDF_Albedo_Parameters_Band{modis_band}"
    token = _login(credentials)
    try:
        task_id = _submit(token, layer, sites, start_date, end_date)
        _wait(token, task_id, timeout_s=timeout_s, poll_s=poll_s)
        rows = _collect(token, task_id, layer)
    finally:
        _logout(token)
    return rows


def _login(credentials: Credentials) -> str:
    """Exchange the Earthdata credentials for a bearer token."""
    if credentials.token:
        logger.info("appeears.login", source=credentials.source, kind="token")
        return credentials.token
    request = urllib.request.Request(
        f"{ROOT}/login",
        data=b"",
        headers={
            "Accept": "application/json",
            **credentials.auth_header(),
        },
        method="POST",
    )
    payload = _read(request, "login")
    token = payload.get("token")
    if not token:
        raise AppeearsError(f"AppEEARS login returned no token: {payload}")
    logger.info(
        "appeears.login",
        source=credentials.source,
        username=credentials.username,
        expires=payload.get("expiration"),
    )
    return str(token)


def _logout(token: str) -> None:
    """Give the token back, so it does not sit valid until it expires."""
    request = urllib.request.Request(
        f"{ROOT}/logout",
        data=b"",
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.debug("appeears.logout_failed", error=str(exc))


def _submit(
    token: str,
    layer: str,
    sites: Sequence[tuple[str, float, float]],
    start_date: str,
    end_date: str,
) -> str:
    """Submit the point-sample task and return its id."""
    task = {
        "task_type": "point",
        "task_name": "adjeff-hotspot-mcd43a1",
        "params": {
            "dates": [
                {
                    "startDate": _to_appeears_date(start_date),
                    "endDate": _to_appeears_date(end_date),
                }
            ],
            "layers": [{"product": PRODUCT, "layer": layer}],
            "coordinates": [
                {"latitude": lat, "longitude": lon, "id": name, "category": "hotspot"}
                for name, lat, lon in sites
            ],
        },
    }
    request = urllib.request.Request(
        f"{ROOT}/task",
        data=json.dumps(task).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    payload = _read(request, "task submission")
    task_id = payload.get("task_id")
    if not task_id:
        raise AppeearsError(f"AppEEARS accepted no task: {payload}")
    logger.info(
        "appeears.submitted",
        task_id=task_id,
        sites=len(sites),
        layer=layer,
        start=start_date,
        end=end_date,
    )
    return str(task_id)


def _wait(token: str, task_id: str, *, timeout_s: float, poll_s: float) -> None:
    """Block until the task is done, saying where it is while it runs."""
    deadline = time.monotonic() + timeout_s
    seen = ""
    while time.monotonic() < deadline:
        payload = _read(
            urllib.request.Request(
                f"{ROOT}/task/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
            ),
            "task status",
        )
        status = str(payload.get("status", "unknown"))
        if status != seen:
            logger.info("appeears.status", task_id=task_id, status=status)
            seen = status
        if status == "done":
            return
        if status in ("error", "failed"):
            raise AppeearsError(
                f"AppEEARS task {task_id} failed: {payload.get('error', payload)}"
            )
        time.sleep(poll_s)
    raise AppeearsError(
        f"AppEEARS task {task_id} was still {seen!r} after {timeout_s:.0f} s. "
        f"It is not lost: check https://appeears.earthdatacloud.nasa.gov and "
        f"raise --appeears-timeout."
    )


def _collect(token: str, task_id: str, layer: str) -> list[dict[str, float | str]]:
    """Download the finished bundle and average each site over the period."""
    bundle = _read(
        urllib.request.Request(
            f"{ROOT}/bundle/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
        ),
        "bundle listing",
    )
    files = bundle.get("files", [])
    wanted = [f for f in files if str(f.get("file_name", "")).endswith(".csv")]
    if not wanted:
        raise AppeearsError(
            f"AppEEARS bundle {task_id} holds no CSV, only "
            f"{[f.get('file_name') for f in files]}"
        )
    # Several CSVs come back; the per-observation one carries the layer
    # name, the others are metadata and statistics.
    chosen = max(wanted, key=lambda f: layer in str(f.get("file_name", "")))
    request = urllib.request.Request(
        f"{ROOT}/bundle/{task_id}/{chosen['file_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AppeearsError(f"AppEEARS bundle download failed: {exc}") from exc

    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            raw = archive.read(name)

    logger.info(
        "appeears.collected", task_id=task_id, file=chosen.get("file_name")
    )
    return _average_sites(raw.decode("utf-8", "replace"), layer)


def _average_sites(text: str, layer: str) -> list[dict[str, float | str]]:
    """Average the three weights per site over every valid date."""
    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    picked = {
        component: _column_for(columns, layer, index)
        for index, component in enumerate(COMPONENTS)
    }

    totals: dict[str, dict[str, float]] = {}
    locations: dict[str, tuple[float, float]] = {}
    for row in reader:
        site = row.get("ID") or row.get("Category") or "unknown"
        values = {c: _value(row.get(col)) for c, col in picked.items()}
        if any(v is None for v in values.values()):
            continue
        bucket = totals.setdefault(site, {c: 0.0 for c in COMPONENTS} | {"n": 0.0})
        for component, value in values.items():
            bucket[component] += float(value)  # type: ignore[arg-type]
        bucket["n"] += 1.0
        locations.setdefault(
            site, (float(row.get("Latitude", 0)), float(row.get("Longitude", 0)))
        )

    rows: list[dict[str, float | str]] = []
    for site, bucket in totals.items():
        n = bucket["n"]
        if n == 0:
            continue
        lat, lon = locations[site]
        rows.append(
            {
                "site": site,
                "lat": lat,
                "lon": lon,
                "f_iso": bucket["iso"] / n,
                "f_vol": bucket["vol"] / n,
                "f_geo": bucket["geo"] / n,
                "n_dates": int(n),
            }
        )
    if not rows:
        raise AppeearsError(
            "AppEEARS returned no usable observation: every date was fill or "
            "quality-flagged. Widen --start-date/--end-date."
        )
    logger.info("appeears.averaged", sites=len(rows))
    return rows


def _column_for(columns: Sequence[str], layer: str, index: int) -> str:
    """Return the column holding component *index* of a 3-D layer.

    Raises
    ------
    AppeearsError
        When it is not there under any spelling seen so far.  A silent
        fallback would hand the study a triplet in the wrong order,
        which no downstream check would catch.
    """
    stem = f"{PRODUCT.replace('.', '_')}_{layer}"
    for candidate in (f"{stem}_{index}", f"{layer}_{index}", f"{stem}_Num_{index}"):
        if candidate in columns:
            return candidate
    raise AppeearsError(
        f"AppEEARS did not return component {index} of {layer}. Columns were "
        f"{list(columns)}. The layer naming has changed and the mapping in "
        f"adjeff_article_1.appeears needs updating."
    )


def _value(raw: str | None) -> float | None:
    """Return a scaled weight, or ``None`` for fill and for nonsense."""
    if raw is None or raw == "":
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    if number == FILL or number < 0:
        return None
    # AppEEARS may already have applied the scale factor; a weight above
    # one is the unscaled integer, below it the physical value.
    return number * SCALE if number > 1.0 else number


def _to_appeears_date(composite: str) -> str:
    """Convert ``AYYYYDDD`` to the ``MM-DD-YYYY`` AppEEARS asks for."""
    text = composite[1:] if composite.startswith("A") else composite
    if len(text) != 7 or not text.isdigit():
        raise AppeearsError(
            f"{composite!r} is not a composite date of the form AYYYYDDD"
        )
    day = datetime(int(text[:4]), 1, 1) + timedelta(days=int(text[4:]) - 1)
    return day.strftime("%m-%d-%Y")


def _read(request: urllib.request.Request, what: str) -> dict:
    """Return the decoded JSON of one call, or say what went wrong."""
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        if exc.code in (401, 403):
            raise AppeearsError(
                f"AppEEARS refused the {what} ({exc.code}). Check the Earthdata "
                f"username and password, and that the account has accepted the "
                f"LP DAAC end-user licence at "
                f"https://appeears.earthdatacloud.nasa.gov. Answer: {detail}"
            ) from exc
        raise AppeearsError(
            f"AppEEARS answered {exc.code} to the {what}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AppeearsError(f"AppEEARS unreachable for the {what}: {exc}") from exc
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        head = payload[:200].decode("utf-8", "replace")
        raise AppeearsError(
            f"AppEEARS did not return JSON for the {what}: {head}"
        ) from exc
    return decoded if isinstance(decoded, dict) else {"payload": decoded}
