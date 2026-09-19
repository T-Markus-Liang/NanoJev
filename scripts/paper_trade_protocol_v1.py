#!/usr/bin/env python3
"""Frozen-protocol loader for the B0 paper-trading replay (task T1, B0-0).

What this is: the artifact that makes a paper-trading number reproducible. A protocol is a
JSON file that pins every run parameter; this module verifies the file's own SHA-256 and
returns the parsed protocol plus the evidence a receipt must carry.

What this is NOT: an authorization system, a trading configuration, or a claim of edge. It
only makes "same input, same result" checkable and "different result" explainable.

Design rules:
* The protocol is authoritative. When a driver is given a protocol, the protocol's values win
  over command-line defaults; a mismatch between the two is an error, not a silent override.
* An unreadable, unparsable, wrong-schema, or hash-mismatched protocol aborts the run.
* Unknown fields are preserved but never interpreted, so a newer protocol can be added without
  silently changing behaviour here.
"""
import hashlib
import json
import pathlib

SCHEMA_VERSION = "nanojev-paper-trade-b0-protocol-v1"

# Fields the driver reads. A protocol missing any of these cannot describe a run.
REQUIRED_POLICY_FIELDS = (
    "half_spread_bps", "fee_bps", "leverage", "margin_mode", "initial_cash",
    "seed", "order_time_to_live_days", "sizing", "capacity", "on_divergence",
    "participation_fraction", "reference_notional_volume",
)
REQUIRED_TOP_FIELDS = ("run_id", "symbols", "first_day", "last_day", "strategy", "policy",
                       "contracts", "input_manifests")

SIZING_MODES = ("fixed_notional",)
CAPACITY_MODES = ("fixed", "venue_volume", "off")
DIVERGENCE_MODES = ("error", "report")
MARGIN_MODES = ("cross", "isolated")


class ProtocolError(Exception):
    """Raised for any unreadable, unparsable, wrong-schema, or hash-mismatched protocol."""


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def expected_digest(protocol_path, explicit=None):
    """The expected protocol digest: an explicit value, else a ``<protocol>.sha256`` sidecar."""
    if explicit is not None:
        return explicit.strip()
    sidecar = pathlib.Path(protocol_path).with_suffix(".sha256")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def load_protocol(path, expected=None):
    """Load and verify a frozen protocol.

    Returns ``(protocol, evidence)``. ``evidence`` carries the values a receipt must record so
    that any later reader can tell which frozen configuration produced the number.
    """
    protocol_path = pathlib.Path(path)
    _require(protocol_path.exists(), f"protocol not found: {protocol_path}")
    raw = protocol_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    expected = expected_digest(protocol_path, expected)
    if expected is not None:
        _require(digest == expected,
                 f"protocol digest mismatch: file is {digest}, expected {expected}")

    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"protocol is not valid JSON: {error}") from error

    _require(isinstance(protocol, dict), "protocol must be a JSON object")
    _require(protocol.get("schema_version") == SCHEMA_VERSION,
             f"unsupported protocol schema: {protocol.get('schema_version')!r} "
             f"(this loader implements {SCHEMA_VERSION})")

    missing = [name for name in REQUIRED_TOP_FIELDS if name not in protocol]
    _require(not missing, f"protocol is missing required fields: {missing}")

    policy = protocol["policy"]
    _require(isinstance(policy, dict), "protocol.policy must be an object")
    missing = [name for name in REQUIRED_POLICY_FIELDS if name not in policy]
    _require(not missing, f"protocol.policy is missing required fields: {missing}")

    _require(policy["sizing"] in SIZING_MODES,
             f"protocol.policy.sizing must be one of {SIZING_MODES}")
    _require(policy["capacity"] in CAPACITY_MODES,
             f"protocol.policy.capacity must be one of {CAPACITY_MODES}")
    _require(policy["on_divergence"] in DIVERGENCE_MODES,
             f"protocol.policy.on_divergence must be one of {DIVERGENCE_MODES}")
    _require(policy["margin_mode"] in MARGIN_MODES,
             f"protocol.policy.margin_mode must be one of {MARGIN_MODES}")
    _require(isinstance(policy["seed"], int) and not isinstance(policy["seed"], bool),
             "protocol.policy.seed must be an integer")
    for name in ("participation_fraction",):
        value = policy[name]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and 0.0 < float(value) <= 1.0,
                 f"protocol.policy.{name} must be in (0, 1]")
    value = policy["reference_notional_volume"]
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0,
             "protocol.policy.reference_notional_volume must be positive")

    strategy = protocol["strategy"]
    _require(isinstance(strategy, dict) and "fast" in strategy and "slow" in strategy,
             "protocol.strategy requires fast and slow")
    _require(protocol["symbols"] and isinstance(protocol["symbols"], list),
             "protocol.symbols must be a non-empty list")
    _require(protocol["contracts"] and isinstance(protocol["contracts"], list),
             "protocol.contracts must be a non-empty list")

    evidence = {"protocol_sha256": digest, "protocol_path": str(protocol_path),
                "protocol_run_id": protocol["run_id"],
                "protocol_verified_against_expected": expected is not None}
    return protocol, evidence


def input_manifest_sha256(protocol, source, data_root):
    """Digest of the fetch manifest named by the protocol for ``source``.

    The manifest is the contract for *which files* fed the run. Hashing it (rather than every
    data file) keeps the receipt small while still failing loudly when the data set changed.
    """
    manifests = protocol["input_manifests"]
    _require(source in manifests, f"protocol has no input manifest for source {source!r}")
    declared = pathlib.Path(manifests[source])
    manifest_path = declared if data_root is None else pathlib.Path(data_root) / declared
    _require(manifest_path.exists(), f"input manifest missing: {manifest_path}")
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
