"""Where the NASA Earthdata credentials come from, and where they do not.

MODIS BRDF products are served from hosts that ask who you are, so the
hotspot figure needs an account.  A repository is the one place those
details must never live: this reads them from outside it, in the order a
reader would expect, and never writes them anywhere.

Order of precedence, first match wins:

1. ``--earthdata-user`` and ``--earthdata-password`` on the command line,
   or ``--earthdata-token``;
2. the environment: ``EARTHDATA_TOKEN``, or ``EARTHDATA_USERNAME`` and
   ``EARTHDATA_PASSWORD``;
3. a TOML file, ``~/.config/adjeff-article/credentials.toml`` by default,
   moved with ``--credentials`` or ``ADJEFF_ARTICLE_CREDENTIALS``;
4. ``~/.netrc``, which is what NASA's own documentation tells you to
   write and what most tools already read.

The file looks like this::

    [earthdata]
    username = "walcark"
    password = "…"
    # or, instead of both:
    token = "…"

and must not be readable by anyone else, which is checked rather than
assumed.
"""

from __future__ import annotations

import netrc
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ._logging import get_logger

__all__ = ["Credentials", "add_credentials_arguments", "load_credentials"]

logger = get_logger(__name__)

#: Host the credentials authenticate against.  NASA's single sign-on
#: covers every DAAC, so one entry serves them all.
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

#: Where the file lives unless told otherwise.  Outside the repository,
#: deliberately: nothing under version control should be able to hold a
#: password by accident.
DEFAULT_PATH = Path.home() / ".config" / "adjeff-article" / "credentials.toml"

#: Environment variable naming another path.
PATH_ENV = "ADJEFF_ARTICLE_CREDENTIALS"


@dataclass(frozen=True)
class Credentials:
    """A username and password, or a bearer token, for Earthdata.

    Attributes
    ----------
    username : str or None
        Earthdata login.
    password : str or None
        Earthdata password.  Never logged, never echoed.
    token : str or None
        Bearer token, which Earthdata issues as an alternative to a
        password and which is the better thing to keep on disk: it can be
        revoked without changing the account.
    source : str
        Where these came from, for a log line that has to say something
        without saying the secret.
    """

    username: str | None = None
    password: str | None = None
    token: str | None = None
    source: str = "nowhere"

    @property
    def available(self) -> bool:
        """Return whether anything usable was found."""
        return bool(self.token or (self.username and self.password))

    def auth_header(self) -> dict[str, str]:
        """Return the Authorization header these credentials imply.

        Returns
        -------
        dict[str, str]
            A single-entry header, or an empty dict when nothing was
            found, so that a caller can splat it either way.
        """
        import base64

        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if self.username and self.password:
            raw = f"{self.username}:{self.password}".encode()
            return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
        return {}

    def __repr__(self) -> str:
        """Return a representation that cannot leak the secret.

        The default dataclass repr prints every field, and this object
        ends up inside tracebacks and debugger frames.
        """
        what = "token" if self.token else "password" if self.password else "nothing"
        return (
            f"Credentials(username={self.username!r}, holds={what}, "
            f"source={self.source!r})"
        )


def add_credentials_arguments(parser: object) -> None:
    """Register the Earthdata options on an argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Notes
    -----
    Passing a password on the command line puts it in the shell history
    and in the process list, where anyone on the machine can read it. The
    option exists because it is occasionally the only way, and the help
    text says so.
    """
    group = parser.add_argument_group(  # type: ignore[attr-defined]
        "Earthdata credentials",
        "Read from the environment or from a file when not given here.",
    )
    group.add_argument("--earthdata-user", help="Earthdata username")
    group.add_argument(
        "--earthdata-password",
        help="Earthdata password (visible in shell history; prefer a file)",
    )
    group.add_argument(
        "--earthdata-token", help="Earthdata bearer token, instead of a password"
    )
    group.add_argument(
        "--credentials",
        type=Path,
        help=f"TOML file holding them (default: {DEFAULT_PATH})",
    )


def load_credentials(args: object = None) -> Credentials:
    """Return the first credentials found, searching four places in order.

    Parameters
    ----------
    args : argparse.Namespace or None
        Parsed arguments, when the caller registered
        :func:`add_credentials_arguments`.  ``None`` skips that source.

    Returns
    -------
    Credentials
        What was found, or an empty set whose ``available`` is ``False``.
        Nothing raises: a figure that can run without them should be able
        to, and the one that cannot says so itself.

    Notes
    -----
    The chosen source is logged; the secret never is.
    """
    found = (
        _from_args(args)
        or _from_environment()
        or _from_file(_config_path(args))
        or _from_netrc()
        or Credentials()
    )
    if found.available:
        logger.info(
            "credentials.loaded",
            source=found.source,
            username=found.username,
            kind="token" if found.token else "password",
        )
    else:
        logger.debug("credentials.absent", looked_in=str(_config_path(args)))
    return found


def _from_args(args: object) -> Credentials | None:
    """Return credentials given on the command line, if any."""
    if args is None:
        return None
    token = getattr(args, "earthdata_token", None)
    user = getattr(args, "earthdata_user", None)
    password = getattr(args, "earthdata_password", None)
    if token:
        return Credentials(username=user, token=token, source="command line")
    if user and password:
        return Credentials(username=user, password=password, source="command line")
    return None


def _from_environment() -> Credentials | None:
    """Return credentials from the environment, if any."""
    token = os.environ.get("EARTHDATA_TOKEN")
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if token:
        return Credentials(username=user, token=token, source="environment")
    if user and password:
        return Credentials(username=user, password=password, source="environment")
    return None


def _config_path(args: object = None) -> Path:
    """Return the credentials file to read, from args, env, or default."""
    given = getattr(args, "credentials", None) if args is not None else None
    if given:
        return Path(given)
    from_env = os.environ.get(PATH_ENV)
    return Path(from_env) if from_env else DEFAULT_PATH


def _from_file(path: Path) -> Credentials | None:
    """Return credentials from a TOML file, if it exists and holds them."""
    if not path.is_file():
        return None
    _warn_if_world_readable(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    section = data.get("earthdata", data)
    token = section.get("token")
    user = section.get("username")
    password = section.get("password")
    source = f"file {path}"
    if token:
        return Credentials(username=user, token=token, source=source)
    if user and password:
        return Credentials(username=user, password=password, source=source)
    logger.warning("credentials.file_incomplete", path=str(path))
    return None


def _from_netrc() -> Credentials | None:
    """Return credentials from ``~/.netrc``, which NASA's own docs ask for."""
    path = Path.home() / ".netrc"
    if not path.is_file():
        return None
    try:
        entry = netrc.netrc(str(path)).authenticators(EARTHDATA_HOST)
    except (netrc.NetrcParseError, OSError) as exc:
        logger.warning("credentials.netrc_unreadable", error=str(exc))
        return None
    if entry is None:
        return None
    login, _, password = entry
    if login and password:
        return Credentials(
            username=login, password=password, source=f"netrc {EARTHDATA_HOST}"
        )
    return None


def _warn_if_world_readable(path: Path) -> None:
    """Say so when a file holding a password is readable by others.

    Not an error: it is the user's machine and their call.  But a
    password file created with the default umask is group and world
    readable, and nobody notices unless told.
    """
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "credentials.file_permissive",
            path=str(path),
            mode=oct(stat.S_IMODE(mode)),
            fix=f"chmod 600 {path}",
        )
