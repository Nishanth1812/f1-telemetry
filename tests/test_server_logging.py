"""The server stays quiet about clients that hang up before the handshake finishes."""

from __future__ import annotations

import logging

from websockets.exceptions import ConnectionClosedError

from f1telemetry.server import _DroppedBeforeHandshake  # pyright: ignore[reportPrivateUsage]


def _handshake_record(error: BaseException) -> logging.LogRecord:
    return logging.LogRecord(
        "websockets.server",
        logging.ERROR,
        __file__,
        1,
        "opening handshake failed",
        None,
        (type(error), error, None),
    )


def test_hangup_before_handshake_is_not_logged_as_an_error() -> None:
    gate = _DroppedBeforeHandshake()
    assert not gate.filter(_handshake_record(ConnectionClosedError(None, None)))


def test_other_handshake_failures_are_still_logged() -> None:
    gate = _DroppedBeforeHandshake()
    assert gate.filter(_handshake_record(ValueError("a real fault")))
