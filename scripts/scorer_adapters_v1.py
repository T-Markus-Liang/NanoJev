#!/usr/bin/env python3
"""Pluggable scorer adapters for the main-model context gateway.

The context gate in :mod:`context_gate_v1` is scorer-agnostic: it hands a caller a
``{"states": [...]}`` payload and validates the returned boolean probability contract.
This module supplies three adapter *shapes* without adding a third-party dependency:

1. :class:`NanoJevHTTPScorer` - the existing local NanoJev decision service contract
   (``POST /api/evaluate``), reusing the literal-loopback client from
   ``context_gate_local.LoopbackPredictor``.
2. :class:`InProcessScorer` - a caller-supplied in-process callable.
3. :class:`LayaEncoderScorerAdapter` - a documented, configurable shape for a
   laya-style local encoder decision service. ``laya`` is **never** installed,
   downloaded, imported, or enabled by this module. The adapter only fixes the HTTP
   contract such a service would have to satisfy, and it refuses to run until an
   operator explicitly points it at a literal loopback origin.

Two decorators are provided for the gateway's fail-open requirements:

* :class:`DeadlineScorer` bounds a scorer call with a wall-clock deadline. Python
  threads cannot be force-killed, so a slow underlying call may keep running after the
  deadline; the deadline only guarantees that the *gateway* stops waiting and fails
  open. This matches the documented caveat that an arbitrary in-process scorer has no
  true cancellation.
* :class:`FailureRecordingScorer` records whether the last failure was a timeout or a
  general exception so the gateway can label a fail-open receipt without copying any
  exception text (which could contain private prompt text).

No API key is required, read, or logged anywhere in this module.
"""

from http.client import HTTPConnection
import json
import math
import socket
import threading
from urllib.parse import urlsplit

from context_gate_local import LoopbackPredictor
from context_gate_v1 import serialized
from predict_toy_decisions import reject_nonfinite, unique_object


SCORER_KINDS = ("none", "http", "laya")
DEFAULT_NANOJEV_URL = "http://127.0.0.1:8765"


class ScorerError(RuntimeError):
    """Adapter is unavailable or the scoring service failed; the gate fails open."""


class ScorerTimeout(TimeoutError):
    """The scorer exceeded its deadline; the gate fails open."""


def _validate_timeout(timeout, upper=3600.0):
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= upper:
        raise ValueError(f"timeout must be finite and in (0,{upper}] seconds")
    return float(timeout)


def _loopback_origin(url):
    """Return (host, port) only for a literal loopback HTTP origin; reject everything else."""
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("only literal loopback HTTP origins are allowed")
    return parsed.hostname, parsed.port or 80


class NanoJevHTTPScorer:
    """Score through the existing local NanoJev service (``POST /api/evaluate``).

    Reuses ``LoopbackPredictor`` verbatim, including its loopback-only origin check,
    request/response byte budgets, no-proxy/no-redirect transport, and bounded socket
    timeout. A connection failure or timeout propagates as an exception and the gate
    fails open.
    """

    def __init__(self, url=DEFAULT_NANOJEV_URL, timeout=5.0):
        self._predict = LoopbackPredictor(url, timeout)
        self.timeout = _validate_timeout(timeout, upper=30.0)

    def __call__(self, payload):
        try:
            return self._predict(payload)
        except (TimeoutError, socket.timeout) as error:
            raise ScorerTimeout("local scorer socket timed out") from error
        except (ValueError, OSError) as error:
            raise ScorerError("local scorer request failed") from error


class InProcessScorer:
    """Adapt a caller-supplied in-process callable to the scorer contract.

    The callable receives the shared scoring payload and must return the existing
    ``{"states": [...]}`` contract. No deadline is imposed here; wrap with
    :class:`DeadlineScorer` when a bounded deadline is required.
    """

    def __init__(self, scorer):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer

    def __call__(self, payload):
        return self.scorer(payload)


class DeadlineScorer:
    """Bound any scorer with a wall-clock deadline; raise :class:`ScorerTimeout` on expiry."""

    def __init__(self, scorer, timeout=5.0):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer
        self.timeout = _validate_timeout(timeout)

    def __call__(self, payload):
        outcome = {}

        def run():
            try:
                outcome["result"] = self.scorer(payload)
            except BaseException as error:  # noqa: BLE001 - forwarded to the caller unchanged
                outcome["error"] = error

        worker = threading.Thread(target=run, name="nanojev-scorer", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            # The call is abandoned, not cancelled; the gateway fails open regardless.
            raise ScorerTimeout("scorer deadline exceeded")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"]


class FailureRecordingScorer:
    """Record the last failure kind (``timeout`` / ``exception``) without its text."""

    def __init__(self, scorer):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer
        self.last_failure = None

    def __call__(self, payload):
        self.last_failure = None
        try:
            return self.scorer(payload)
        except (TimeoutError, socket.timeout):
            self.last_failure = "timeout"
            raise
        except BaseException:  # noqa: BLE001 - re-raised for the gate's fail-open path
            self.last_failure = "exception"
            raise


class _LoopbackJSONPoster:
    """Minimal loopback-only JSON POST client with a configurable path.

    Mirrors ``LoopbackPredictor``'s transport guarantees (no environment proxies, no
    redirect following, bounded timeout) but allows an adapter-specific endpoint path.
    """

    def __init__(self, url, path, timeout):
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("endpoint path must start with '/'")
        self.host, self.port = _loopback_origin(url)
        self.path = path
        self.timeout = _validate_timeout(timeout, upper=30.0)
        self.max_bytes = 2_000_000

    def __call__(self, payload):
        body = serialized(payload).encode("utf-8")
        if len(body) > self.max_bytes:
            raise ScorerError("scoring request budget exceeded")
        connection = HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("POST", self.path, body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ScorerError("scoring service returned a non-200 status")
            content = response.read(self.max_bytes + 1)
            if len(content) > self.max_bytes:
                raise ScorerError("scoring response budget exceeded")
            return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
        except (TimeoutError, socket.timeout) as error:
            raise ScorerTimeout("scoring service timed out") from error
        except (ValueError, OSError) as error:
            raise ScorerError("scoring service request failed") from error
        finally:
            connection.close()


class LayaEncoderScorerAdapter:
    """Documented shape for a laya-style local encoder decision service. Not installed.

    ``laya`` (a 421M ModernBERT encoder decision project) is an external, optional
    encoder-architecture candidate for the Track A gate. This adapter deliberately does
    **not** import, install, download, or call it. It only defines what an operator would
    have to provide to use such a service as the gate scorer:

    * a literal loopback HTTP origin (e.g. ``http://127.0.0.1:8791``),
    * an ``/api/evaluate``-compatible JSON endpoint accepting the shared
      ``{"states": [...]}`` payload (optionally with a ``model`` hint),
    * a response that satisfies the same validated boolean-probability contract.

    Until ``url`` is explicitly configured the adapter raises :class:`ScorerError`, so an
    unconfigured laya route can only ever fail open.
    """

    def __init__(self, url=None, timeout=5.0, endpoint="/api/evaluate", model_id=None):
        self.endpoint = endpoint
        self.model_id = model_id
        self._poster = _LoopbackJSONPoster(url, endpoint, timeout) if url else None

    @property
    def available(self):
        return self._poster is not None

    def __call__(self, payload):
        if self._poster is None:
            raise ScorerError("laya-style encoder adapter is not configured")
        request_payload = payload
        if self.model_id is not None:
            request_payload = {**payload, "model": self.model_id}
        return self._poster(request_payload)


def build_scorer(kind="none", url=None, timeout=5.0, inprocess=None,
                 endpoint="/api/evaluate", model_id=None):
    """Build a scorer adapter by name.

    ``none`` returns ``None`` (the gate then reports ``scorer_unavailable`` and retains
    every segment). ``http`` builds :class:`NanoJevHTTPScorer`. ``laya`` builds the
    documented, unconfigured-unless-url-given :class:`LayaEncoderScorerAdapter`.
    ``inprocess`` builds :class:`InProcessScorer` from ``inprocess``.
    """
    if kind == "none":
        return None
    if kind == "http":
        return NanoJevHTTPScorer(url or DEFAULT_NANOJEV_URL, timeout)
    if kind == "laya":
        return LayaEncoderScorerAdapter(url=url, timeout=timeout, endpoint=endpoint, model_id=model_id)
    if kind == "inprocess":
        return InProcessScorer(inprocess)
    raise ValueError("unknown scorer kind")
