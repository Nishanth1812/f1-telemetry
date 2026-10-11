"""Caller-supplied run manifest for P6-T7.

A record of the five facts a run cites: seed, car-spec version, scenario version,
setup hash, and git SHA. Every value is supplied by the caller; this module only
validates, serialises deterministically, and parses back. It does not infer any
value (no setup hashing, no car_spec.yaml or scenario resolution, no git calls)
and claims no automatic reproducibility.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

__all__ = [
    "MANIFEST_METADATA_KEY",
    "MANIFEST_VERSION",
    "RunManifest",
    "from_metadata",
]

MANIFEST_VERSION: Final[int] = 1
MANIFEST_METADATA_KEY: Final[str] = "f1telemetry/run_manifest"

_HEX_RE: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]+\Z")
_GIT_SHA_LENGTHS: Final[frozenset[int]] = frozenset({40, 64})


def _error(field: str, detail: str) -> ValueError:
    return ValueError(f"run manifest {field}: {detail}")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _error(field, f"must be a string, got {type(value).__name__}")
    if not value or value != value.strip():
        raise _error(field, "must be a non-empty, unpadded string")
    return value


def _hex(value: object, field: str) -> str:
    text = _text(value, field)
    if not _HEX_RE.match(text):
        raise _error(field, "must be lower-case hex")
    return text


@dataclass(frozen=True, slots=True)
class RunManifest:
    seed: int
    car_spec_version: str
    scenario_version: str
    setup_hash: str
    git_sha: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise _error("seed", "must be a non-negative integer")
        object.__setattr__(
            self, "car_spec_version", _text(self.car_spec_version, "car_spec_version")
        )
        object.__setattr__(
            self, "scenario_version", _text(self.scenario_version, "scenario_version")
        )
        object.__setattr__(self, "setup_hash", _hex(self.setup_hash, "setup_hash"))
        git_sha = _hex(self.git_sha, "git_sha")
        if len(git_sha) not in _GIT_SHA_LENGTHS:
            raise _error("git_sha", "must be a 40- or 64-hex-character commit id")
        object.__setattr__(self, "git_sha", git_sha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "seed": self.seed,
            "car_spec_version": self.car_spec_version,
            "scenario_version": self.scenario_version,
            "setup_hash": self.setup_hash,
            "git_sha": self.git_sha,
        }

    def canonical(self) -> str:
        """Compact, key-sorted JSON: the one deterministic serialisation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, document: object) -> RunManifest:
        """Validate a decoded manifest document. Takes ``object`` so the mapping check is real."""
        if not isinstance(document, Mapping):
            raise _error("document", f"expected a mapping, got {type(document).__name__}")
        if any(not isinstance(key, str) for key in document):
            raise _error("document", "field names must be strings")
        expected = {
            "manifest_version",
            "seed",
            "car_spec_version",
            "scenario_version",
            "setup_hash",
            "git_sha",
        }
        unknown = set(document) - expected
        missing = expected - set(document)
        if unknown:
            raise _error("document", f"unknown field(s) {sorted(unknown)}")
        if missing:
            raise _error("document", f"missing field(s) {sorted(missing)}")
        # Every value is validated by __post_init__, so hand them over untyped on purpose.
        fields = cast("Mapping[str, Any]", document)
        version = fields["manifest_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != MANIFEST_VERSION:
            raise _error("manifest_version", f"unsupported version {version!r}")
        return cls(
            seed=fields["seed"],
            car_spec_version=fields["car_spec_version"],
            scenario_version=fields["scenario_version"],
            setup_hash=fields["setup_hash"],
            git_sha=fields["git_sha"],
        )

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise _error("document", f"is not valid JSON: {exc}") from exc
        except RecursionError as exc:
            # Nesting deeper than the parser's own recursion limit is still an
            # unreadable document, so it stays inside the ValueError contract.
            raise _error("document", "is nested too deeply to parse") from exc
        return cls.from_dict(document)

    def to_metadata(self) -> dict[str, str]:
        """Parquet key-value metadata: the manifest under one namespaced key."""
        return {MANIFEST_METADATA_KEY: self.canonical()}


def from_metadata(metadata: Mapping[Any, Any] | None) -> RunManifest | None:
    """Read a manifest back out of Arrow/Parquet key-value metadata.

    Absent key returns ``None`` (normal for files written before this hook);
    a present but unreadable manifest is an error, not a silent absence.
    """
    if metadata is None:
        return None
    raw = metadata.get(MANIFEST_METADATA_KEY.encode("utf-8"))
    if raw is None:
        raw = metadata.get(MANIFEST_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeError as exc:
        raise _error(MANIFEST_METADATA_KEY, f"is not valid UTF-8: {exc}") from exc
    if not isinstance(payload, str):
        raise _error(MANIFEST_METADATA_KEY, f"must be text, got {type(raw).__name__}")
    return RunManifest.from_json(payload)
