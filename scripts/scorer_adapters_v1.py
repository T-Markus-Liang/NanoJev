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
* :class:`FailureRecordingScorer` records whether the latest failed call was a timeout
  or a general exception. A later successful batch does not erase it. The gateway
  creates one recorder per request, so failure information cannot cross requests.
  No exception text (which could contain private prompt text) is copied.

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


SCORER_KINDS = ("none", "http", "laya", "systemone", "cascade")
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
    """Record the latest failure across calls; use a fresh instance per request."""

    def __init__(self, scorer):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer
        self.last_failure = None

    def __call__(self, payload):
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


class SystemOneHTTPScorer:
    """Translate the gate's states payload into local `/v1/systemone` calls.

    The context gate batches candidate states under the NanoJev
    ``{"states": [...]}`` contract. SystemOne-compatible servers such as Winnow
    accept one state plus a named question map per call, so this adapter performs
    one bounded loopback request per candidate state and translates `noul` back to
    the gate's boolean `{false,true}` probability contract. It never sends API keys
    and only permits literal loopback origins through ``_LoopbackJSONPoster``.
    """

    def __init__(self, url, timeout=5.0, endpoint="/v1/systemone", model_id=None):
        self.endpoint = endpoint
        self.model_id = model_id
        self._poster = _LoopbackJSONPoster(url, endpoint, timeout)

    @staticmethod
    def _state_value(value):
        if isinstance(value, str):
            try:
                return json.loads(value, object_pairs_hook=unique_object,
                                  parse_constant=reject_nonfinite)
            except (ValueError, TypeError):
                return value
        return value

    @staticmethod
    def _question_map(questions):
        if not isinstance(questions, dict) or not questions:
            raise ScorerError("systemone scorer requires a non-empty question map")
        converted = {}
        for name, question in questions.items():
            if not isinstance(question, dict):
                raise ScorerError("systemone scorer received an invalid question")
            converted_question = dict(question)
            if converted_question.get("type") == "boolean":
                converted_question["type"] = "noul"
            converted[name] = converted_question
        return converted

    @staticmethod
    def _answer(answer):
        if not isinstance(answer, dict):
            raise ScorerError("systemone scorer returned an invalid answer")
        if answer.get("type") == "noul" and type(answer.get("noul")) in {int, float}:
            probability = float(answer["noul"])
            return {"type": "boolean", "probabilities": {"false": 1.0 - probability,
                                                         "true": probability}}
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict) and {"false", "true"} <= set(probabilities):
            return {"type": "boolean", "probabilities": {"false": float(probabilities["false"]),
                                                         "true": float(probabilities["true"])}}
        raise ScorerError("systemone scorer returned an unsupported answer shape")

    def __call__(self, payload):
        states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(states, list):
            raise ScorerError("systemone scorer requires a states list")
        output_states, usages, models = [], [], set()
        for state in states:
            if not isinstance(state, dict) or not isinstance(state.get("id"), str):
                raise ScorerError("systemone scorer received an invalid state")
            request = {"state": self._state_value(state.get("state")),
                       "questions": self._question_map(state.get("questions"))}
            if self.model_id is not None:
                request["model"] = self.model_id
            response = self._poster(request)
            answers = response.get("answers") if isinstance(response, dict) else None
            if not isinstance(answers, dict):
                raise ScorerError("systemone scorer returned no answers")
            output_states.append({"id": state["id"], "answers": {
                name: self._answer(answer) for name, answer in answers.items()}})
            if isinstance(response.get("usage"), dict):
                usages.append(response["usage"])
            if isinstance(response.get("model"), str):
                models.add(response["model"])
        result = {"checkpoint": {"adapter": "systemone-http", "endpoint": self.endpoint,
                                 "models": sorted(models)},
                  "states": output_states}
        if usages:
            result["usage"] = {"requests": len(usages),
                               "input_tokens": sum(int(u.get("input_tokens", 0)) for u in usages),
                               "output_tokens": sum(int(u.get("output_tokens", 0)) for u in usages)}
        return result


class CascadeScorer:
    """Small-model-first scorer with a stronger fallback for uncertain answers.

    ``fast`` and ``strong`` are ordinary gate scorers. Each candidate state is sent
    to ``fast`` first. If every returned answer has a confident probability
    (``max(probabilities) >= fast_threshold``), the fast answer is retained. If any
    answer is uncertain or malformed, only that state is routed to ``strong``.
    A fast-path exception also routes that state to ``strong`` by default; a strong
    failure propagates so the gate can fail open.
    """

    def __init__(self, fast, strong, fast_threshold=0.95):
        if not callable(fast) or not callable(strong):
            raise TypeError("cascade scorers must be callable")
        if type(fast_threshold) not in {int, float} or not math.isfinite(fast_threshold) or not 0.5 < fast_threshold <= 1:
            raise ValueError("fast_threshold must be finite and in (0.5,1]")
        self.fast = fast
        self.strong = strong
        self.fast_threshold = float(fast_threshold)

    def _confident(self, state):
        answers = state.get("answers") if isinstance(state, dict) else None
        if not isinstance(answers, dict) or not answers:
            return False
        for answer in answers.values():
            probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
            if not isinstance(probabilities, dict) or not probabilities:
                return False
            values = [value for value in probabilities.values() if type(value) in {int, float}]
            if len(values) != len(probabilities) or not values or max(values) < self.fast_threshold:
                return False
        return True

    def __call__(self, payload):
        states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(states, list):
            raise ScorerError("cascade scorer requires a states list")
        output_states, fast_paths, strong_paths, fast_errors = [], [], [], 0
        for state in states:
            if not isinstance(state, dict):
                raise ScorerError("cascade scorer received an invalid state")
            single = {**payload, "states": [state]}
            selected, path = None, "strong"
            try:
                candidate = self.fast(single)
                candidate_state = (candidate.get("states") or [None])[0] if isinstance(candidate, dict) else None
                if self._confident(candidate_state):
                    selected, path = candidate_state, "fast"
            except Exception:
                fast_errors += 1
            if selected is None:
                candidate = self.strong(single)
                candidate_state = (candidate.get("states") or [None])[0] if isinstance(candidate, dict) else None
                if not isinstance(candidate_state, dict):
                    raise ScorerError("cascade strong scorer returned no state")
                selected = candidate_state
            selected = dict(selected)
            selected["id"] = state.get("id")
            output_states.append(selected)
            (fast_paths if path == "fast" else strong_paths).append(state.get("id"))
        return {"checkpoint": {"adapter": "cascade", "fast_threshold": self.fast_threshold,
                               "fast_paths": fast_paths, "strong_paths": strong_paths,
                               "fast_errors": fast_errors},
                "states": output_states}


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
                 endpoint="/api/evaluate", model_id=None, strong_url=None,
                 strong_model_id=None, fast_threshold=0.95):
    """Build a scorer adapter by name.

    ``none`` returns ``None`` (the gate then reports ``scorer_unavailable`` and retains
    every segment). ``http`` builds :class:`NanoJevHTTPScorer`. ``systemone`` builds
    :class:`SystemOneHTTPScorer` for local `/v1/systemone` services. ``cascade`` builds
    :class:`CascadeScorer` with ``url`` as the fast `/v1/systemone` service and
    ``strong_url`` as the strong one. ``laya`` builds the documented,
    unconfigured-unless-url-given :class:`LayaEncoderScorerAdapter`.
    ``inprocess`` builds :class:`InProcessScorer` from ``inprocess``.
    """
    if kind == "none":
        return None
    if kind == "http":
        return NanoJevHTTPScorer(url or DEFAULT_NANOJEV_URL, timeout)
    if kind == "systemone":
        if url is None:
            raise ValueError("systemone scorer requires --scorer-url")
        return SystemOneHTTPScorer(url, timeout=timeout, endpoint=endpoint,
                                   model_id=model_id)
    if kind == "cascade":
        if url is None or strong_url is None:
            raise ValueError(
                "cascade scorer requires --scorer-url (fast) and --scorer-strong-url")
        systemone = endpoint if endpoint != "/api/evaluate" else "/v1/systemone"
        return CascadeScorer(
            SystemOneHTTPScorer(url, timeout=timeout, endpoint=systemone,
                                model_id=model_id),
            SystemOneHTTPScorer(strong_url, timeout=timeout, endpoint=systemone,
                                model_id=strong_model_id),
            fast_threshold=fast_threshold)
    if kind == "laya":
        return LayaEncoderScorerAdapter(url=url, timeout=timeout, endpoint=endpoint, model_id=model_id)
    if kind == "inprocess":
        return InProcessScorer(inprocess)
    raise ValueError("unknown scorer kind")
