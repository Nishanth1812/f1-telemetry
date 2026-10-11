default:
    @echo "recipes: install lint type test test-invariant test-golden codegen codegen-check web-build check serve dev"

install:
    uv sync --frozen

lint:
    uv run --frozen ruff check src tests
    uv run --frozen ruff format --check src tests

type:
    uv run --frozen basedpyright

test:
    uv run --frozen pytest

test-invariant:
    uv run --frozen pytest -m invariant

test-golden:
    uv run --frozen pytest -m golden

codegen:
    uv run --frozen f1-codegen

codegen-check:
    uv run --frozen f1-check-contract

web-build:
    npm ci --prefix web
    npm run build --prefix web

# P0 demo: serve synthetic contract telemetry at ws://localhost:8765/ws for the
# dashboard. Ctrl-C shuts it down cleanly. Arguments pass straight through, so
# `just serve --port 9000` or `just serve --seed 7 --hz 10` both work.
serve *ARGS:
    uv run --frozen f1-serve {{ARGS}}

dev:
    npm --prefix web run dev

check: lint type test codegen-check web-build
