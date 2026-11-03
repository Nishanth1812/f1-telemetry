FROM node:24-bookworm-slim@sha256:c86ecead9e55a5210ac1727de38a4ca840bf03dcaf34dc9749addbf383b1920c AS dashboard
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58
WORKDIR /app
ENV UV_LINK_MODE=copy NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
COPY channels.yaml car_spec.yaml ./
COPY tracks/ ./tracks/
COPY examples/ ./examples/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY --from=dashboard /web/dist ./web/dist/
ARG REVISION
ENV F1_TELEMETRY_REVISION=${REVISION}
USER 65534:65534
EXPOSE 8765
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=2)"]
ENTRYPOINT ["/app/.venv/bin/python", "-m", "f1telemetry.demo"]
CMD ["--host", "0.0.0.0", "--allowed-origin", "http://localhost:8765"]
