"""URI read/write helpers (cribbed from the coworld game-server template).

Scheme dispatch on ``file://`` / bare path / ``http(s)://``. No ``s3://`` — hosted
storage arrives as presigned https. A non-default User-Agent is set because the
platform's edge blocks urllib's default UA.
"""

from __future__ import annotations

import gzip
import os
import zlib
from pathlib import Path
from typing import Literal, cast
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

HTTP_USER_AGENT = "coworld-gnomic/0.1"


def maybe_decompress(data: bytes) -> bytes:
    """Transparently inflate zlib- or gzip-compressed bytes.

    Hosted coworld stores the replay zlib-compressed (``replay.z``) and passes it
    straight back via ``COGAME_LOAD_REPLAY_URI``; local runs keep it raw."""
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:  # gzip
        return gzip.decompress(data)
    if len(data) >= 2 and data[0] == 0x78:  # zlib
        try:
            return zlib.decompress(data)
        except zlib.error:
            return data
    return data


def read_data(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        request = Request(uri, headers={"User-Agent": HTTP_USER_AGENT})
        with urlopen(request, timeout=30) as response:
            return response.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    if parsed.scheme == "":
        return Path(uri).read_bytes()
    raise ValueError(f"Unsupported URI for read_data: {uri}")


def artifact_method(env_var: str) -> Literal["POST", "PUT"]:
    method = os.environ.get(env_var, "PUT").upper()
    if method not in {"POST", "PUT"}:
        raise ValueError(f"{env_var} must be PUT or POST")
    return cast("Literal['POST', 'PUT']", method)


def write_data(uri: str, data: bytes | str, *, content_type: str, http_method: str = "PUT") -> None:
    if isinstance(data, str):
        data = data.encode()
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        request = Request(uri, data=data, method=http_method)
        request.add_header("Content-Type", content_type)
        request.add_header("User-Agent", HTTP_USER_AGENT)
        with urlopen(request, timeout=60):
            return
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if parsed.scheme == "":
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    raise ValueError(f"Unsupported URI for write_data: {uri}")
