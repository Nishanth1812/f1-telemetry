"""Built dashboard and bounded operational status on the WebSocket listener."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from urllib.parse import unquote, urlsplit

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response


class WebSite:
    def __init__(
        self,
        root: Path | None,
        status: Callable[[], dict[str, object]],
        *,
        read_only: bool = False,
    ) -> None:
        self.root: Path | None = root.resolve() if root is not None else None
        self.status: Callable[[], dict[str, object]] = status
        self.read_only: bool = read_only
        if self.root is not None and not (self.root / "index.html").is_file():
            raise ValueError(
                f"Built dashboard not found in {self.root}; run npm run build --prefix web"
            )

    async def process_request(
        self, _connection: ServerConnection | None, request: Request
    ) -> Response | None:
        path = unquote(urlsplit(request.path).path)
        if path == "/healthz":
            status = {**self.status(), "read_only": self.read_only}
            code = HTTPStatus.OK if status.get("ready") else HTTPStatus.SERVICE_UNAVAILABLE
            return _response(
                code, json.dumps(status, separators=(",", ":")).encode(), "application/json"
            )
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        if self.root is not None:
            target = (self.root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if target.is_relative_to(self.root) and target.is_file():
                content = await asyncio.to_thread(target.read_bytes)
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                return _response(HTTPStatus.OK, content, content_type)
        return _response(HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain")


def _response(code: HTTPStatus, body: bytes, content_type: str) -> Response:
    return Response(
        code.value,
        code.phrase,
        Headers(
            {
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            }
        ),
        body,
    )
