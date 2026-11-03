import asyncio
from pathlib import Path

from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from f1telemetry.web_site import WebSite


def test_site_confines_paths_and_reports_readiness(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Telemetry</h1>", encoding="utf-8")
    outside = tmp_path.parent / "private-file.txt"
    outside.write_text("private", encoding="utf-8")
    site = WebSite(tmp_path, lambda: {"ready": True, "clients": 0}, read_only=True)

    async def request(path: str) -> Response | None:
        return await site.process_request(None, Request(path, Headers()))

    home = asyncio.run(request("/"))
    assert home is not None and home.status_code == 200
    assert b"Telemetry" in home.body
    health = asyncio.run(request("/healthz"))
    assert health is not None and b'"read_only":true' in health.body
    for path in ("/../private-file.txt", "/%2e%2e/private-file.txt", "/missing.js"):
        result = asyncio.run(request(path))
        assert result is not None and result.status_code == 404
        assert b"private" not in result.body


def test_unready_site_returns_unavailable() -> None:
    site = WebSite(None, lambda: {"ready": False})
    response = asyncio.run(site.process_request(None, Request("/healthz", Headers())))
    assert response is not None and response.status_code == 503
