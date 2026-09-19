# Reversible filtering V1

Status: **reversible half of the decision/compression layer implemented and unit-tested.
Strict fallback (fail-open) was already implemented and tested on 12 failure paths; this
milestone adds the missing reversible half: an applied reduction now carries a
content-free restore manifest and a caller-driven, hash-proven restore path. Active
filtering remains disabled by default. No real provider was called. No latency or
production token-saving claim is made.**

This document specifies the reversibility contract of the NanoJev decision/compression
layer that sits in front of main models. It is the companion to
[Main-model gateway V1](MAIN_MODEL_GATEWAY_V1.md) (transport, shadow/active modes, kill
switch, receipts, token accounting) and [Context gate shadow V1](CONTEXT_SHADOW_V1.md)
(the byte-preserving shadow core). It does not replace or lower either document's gates.

## 1. Files

| File | Role |
|---|---|
| `scripts/context_restore_v1.py` | Restore manifests, caller-driven reconstruction, round-trip verification |
| `scripts/test_context_restore_v1.py` | unittest coverage: byte-exact round-trips, content-free manifests, tamper and malformed-input rejection |
| `scripts/main_model_gateway_v1.py` | **Additive:** emits the restore manifest to the caller as a response header when a reduction is actually applied |
| `scripts/test_main_model_gateway_v1.py` | **Extended:** the header is emitted only for an applied reduction and never on any fail-open path |
| `docs/REVERSIBLE_FILTERING_V1.md` | This specification |

Nothing else in the repository is modified by this milestone. `context_gate_v1.py`,
`context_gate_local.py`, and the shadow documentation are reused unchanged, not forked.

## 2. The reversibility contract

A reduction is **reversible** when, and only when, all of the following hold:

1. **The removed segment is a single text segment.** V1 only ever drops an eligible
   assistant text segment: either a whole assistant message whose `content` is a string, or
   one text part of an assistant list-content message. A message carrying mixed content
   (tool blocks, multiple non-text parts) is not a single-text segment, so a whole-message
   drop of it is rejected outright (`whole_message_drop_requires_string_content`) rather
   than half-restored. Partial drops of its text parts remain reversible.
2. **The pointer is inside the allowed grammar.** Only
   `/{messages|input}/{i}/content` and `/{messages|input}/{i}/content/{j}/text`, with the
   message key matching the wire format, are accepted. Anything else — a different key, a
   role/field pointer, a negative, non-integer, or out-of-range index, an empty pointer —
   is rejected with a fixed reason code.
3. **No position is duplicated or conflicting.** The same pointer twice, two records
   claiming the same position, or a whole-message record on a message that also has
   part records are all rejected. A plan is never partially applied.
4. **The caller supplies the removed content it retained.** `restore_request(reduced,
   manifest, removed_segments)` receives the removed text; each value is checked against
   the manifest's recorded hash and role, then inserted at the recorded position. The
   pointer set must match the manifest exactly: unknown, missing, and extra pointers all
   fail.
5. **The reconstruction is proven, not assumed.** Three independent checks must pass
   before bytes are returned: (a) the reconstruction re-serialized canonically hashes to
   the manifest's `restored_bytes_sha256`; (b) re-applying the recorded reduction to the
   reconstruction reproduces the caller's reduced bytes *byte-for-byte*; (c) the shared
   core's `parse_segments` accepts the reconstruction. Any failure raises
   `RestoreError`; a wrong reconstruction is never returned.

### Byte-exactness, stated precisely

Restore reconstructs the original request *object* and returns its canonical serialization
(UTF-8 JSON, `ensure_ascii=False`, `separators=(",", ":")`, key order preserved, non-finite
constants rejected). Two hashes in the manifest make the guarantee exact instead of
assumed:

* `original_request_sha256` — the exact bytes the caller sent. It identifies the request
  the manifest belongs to.
* `restored_bytes_sha256` — the canonical serialization of that same request, i.e. exactly
  what `restore_request` returns.

When the original body was already canonical (`canonical_json: true`), the two hashes are
equal and the round-trip is **byte-identical** to the original. When a client sends an
indented or otherwise non-canonical body, the two hashes differ: the reconstruction is
byte-identical to the *canonical form* of the original and `verify_round_trip` reports
`restored_bytes_identical: false` rather than claiming identity. No content is lost,
duplicated, or reordered in either case — the reconstructed object re-serializes to
`restored_bytes_sha256`, and the re-reduction check reproduces the reduced bytes exactly.

`verify_round_trip(raw, reduced, manifest, removed_segments=None)` returns a structured
result (`original_request_sha256_matches`, `reduced_request_sha256_matches`,
`canonical_original`, `restored_bytes_identical`, `reconstructed_sha256`, ...) and raises on
tamper or mismatch. Without `removed_segments` it verifies the hashes only and explicitly
reports `reconstruction_performed: false`; it never implies a reconstruction it did not do.

## 3. Who holds what

| Party | Holds | Never holds |
|---|---|---|
| **Caller** | The original request, and the removed segment content it retained | — |
| **Gateway** | Pointers, wire format, segment/role hashes, request hashes, positions | Removed segment text, prompt text, tool output, credentials |
| **Receipts** | The same content-free gate fields as before; no manifest text and no prompt text | Removed segment text |
| **Response header** | `x-nanojev-restore-manifest`: the same content-free manifest | Any raw prompt text |
| **Upstream provider** | Only the reduced bytes (and the caller's own provider request when no reduction applied) | Nothing new; the restore manifest is never forwarded |

The manifest is a *description of a hole*, not the contents of the hole. A manifest without
the caller's retained segments restores nothing.

## 4. Why the gateway must never persist removed text

The gateway is a transport in the middle of a request path that may carry private prompts,
credentials-adjacent context, and regulated data. If it persisted removed segments it would
become a second, increasingly complete copy of client conversations — exactly the copy a
caller sends to a provider under its own data policy. Three properties follow from not
persisting it:

* **No new data at rest.** The removed text only ever exists in the caller's own process.
  Deletion in the reduced request is therefore not a claim that the gateway has «seen and
  stored» the text; it has hashed it and discarded it.
* **No new exfiltration surface.** Receipts and headers are written at rest and traverse
  logs and tracing. Because only hashes and pointers flow there, a leaked receipt or header
  does not leak prompt content.
* **Honest audit.** Because the gateway cannot reconstruct anything on its own, a
  reconstruction claim can only ever be made by the party that actually held the content —
  the caller. That is the same party that can be asked for the original.

A manifest is *not* encryption and not anonymization: a hash of a short or guessable
segment can be dictionary-attacked, and the pointer structure reveals which messages were
dropped. Treat a manifest as being exactly as sensitive as a digest of the request, and
apply the same access controls as to receipts.

## 5. Gateway integration

The gateway change is additive and applies only in active mode when a reduction is actually
sent. In `ContextGateGateway.handle`:

1. The removal plan is applied and re-validated with the core parser, exactly as before.
2. `build_restore_manifest(raw, wire_format, applied_pointers)` is now called for that exact
   plan. If it raises — a plan whose segment is not a single text segment, an
   out-of-grammar pointer, and so on — the request **fails open** with
   `forward_reason: reduction_error` and the original bytes are forwarded unchanged. A
   reduction the gateway cannot describe reversibly is a reduction it does not send. This
   makes `reduction_error` reachable for a new cause (a plan that is applicable but not
   reversible, such as a whole-message drop of a list-content message) while leaving the
   fixed reason-code set, thresholds, modes, and receipt schema unchanged. Partial-content
   drops of a list-content message remain reversible and are still applied.
3. On success the manifest is attached to the **response** as
   `x-nanojev-restore-manifest` (compact JSON). It is emitted only in the branch where
   `forward_reason` is `active_reduced`.
4. The header is never added to the upstream request; the existing
   `x-nanojev-*` stripping behaviour for upstream requests is unchanged. Header emission
   does not change any default, threshold, mode, receipt field, or fail-open reason code.

The call sequence for the caller is:

```text
caller --(original request)--> gateway --(reduced request)--> provider
caller <--(provider response + x-nanojev-restore-manifest)-- gateway

# later, offline, with the segments the caller retained:
restored = restore_request(reduced_bytes_the_gateway_sent, manifest, retained_segments)
assert verify_round_trip(original_bytes, reduced_bytes, manifest, retained_segments)
```

No header is emitted by shadow mode, the kill switch, an unsupported method or wire format,
a protected/unknown drop in the plan, a scorer error/timeout/absence, an uncertain or
malformed score, or a reduction error. Those paths forward the original bytes unchanged, so
there is nothing to restore.

## 6. How strict fallback interacts with reversibility

Strict fallback and reversibility are two halves of one rule, and they compose without
weakening each other:

* **Strict fallback decides whether a reduction may happen at all.** Any doubt — gate
  exception, scorer failure or timeout, malformed/partial/duplicate/nonfinite/non-unit or
  uncertain score, protected segment in the removal set, reduction error, kill switch —
  forwards the ORIGINAL bytes. Fail-open is unchanged and untuned.
* **Reversibility gates the one path that survives fallback.** A reduction that is allowed
  to proceed must additionally be describable by a content-free restore manifest; if it is
  not, the gateway falls back. This makes «reversible» a *precondition of applying*, not a
  post-hoc property.
* **Restore itself fails closed.** Every malformed, tampered, out-of-range, duplicated,
  unknown, or mismatched manifest/segment raises `RestoreError`. There is no «best effort»
  reconstruction.
* **Fail-open never needs restoration.** When the original bytes are forwarded unchanged,
  the caller already has exactly what was sent; no manifest is emitted and none is needed.
* **Reversibility does not relax any gate.** A reversible reduction is still a reduction:
  the Track A acceptance gates, the paired downstream-quality comparisons, and the
  protected-segment zero-deletion requirement all still apply before any production
  enablement.

## 7. What is NOT established

* **No real provider was called.** All evidence is unit tests: the gateway suite uses a
  loopback fake upstream and a loopback fake scorer, and the restore suite is pure
  in-process byte manipulation.
* **No end-to-end reversibility measurement through a real client.** No real caller was
  wired to retain segments and re-issue a restored request; only the library and gateway
  contracts are unit-tested.
* **No latency claim.** Manifest construction and header emission were not benchmarked.
  The existing gate latency measurements from the shadow work are unaffected by this change
  but are not evidence for it.
* **No performance or memory claim.** Manifest size grows with the number of removed
  segments (one record per segment); no size budget, compression, or header-limit test was
  performed, and a very large removal plan could exceed a proxy's header size limit.
* **No cryptographic protection of the manifest.** Hashes are unkeyed SHA-256 and pointers
  are cleartext; dictionary attacks on short segments and inference from pointer structure
  are possible. There is no HMAC, no signing, and no encryption.
* **No guarantee for non-canonical originals beyond canonical equivalence.** A caller that
  sent an indented body gets back the canonical equivalent, not its original bytes; the
  manifest reports this via `canonical_json` and `verify_round_trip` reports
  `restored_bytes_identical: false`.
* **No coverage beyond assistant text.** User messages, tool results, files, retrieval
  snippets, images, and server-side conversation references are not removable in V1 and
  therefore not restorable.
* **No manifest versioning migration.** Only schema
  `nanojev-context-restore-v1` is accepted; an unknown schema version is rejected rather
  than migrated.
* **No production enablement.** Shadow mode remains the default, active filtering remains
  disabled by default, and the kill switch behaviour is unchanged. Enabling production
  active filtering still requires the full Track A acceptance gates plus paired
  downstream-quality and net-token-cost evidence across at least three main-model families
  and a protected-segment stress set with zero deletions.
* **No independent review and no `results/` receipt.** These files have not been reviewed by
  the project's independent reviewer, and this milestone produced no receipt under
  `results/`.

## 8. Reproduction

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_context_restore*.py'
.venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway*.py'
```

The restore suite covers: byte-identical round-trips for all three wire formats and for both
whole-message and partial-content removal (including a message emptied by removing all of its
parts and a mixed whole+partial plan); a no-op round-trip; content-free manifests (no raw
text, only a recorded safe vocabulary, pointers, and hashes); tampered manifest, tampered
segment, tampered role, tampered position, tampered hash/count/flag/schema, and unknown or
missing fields; malformed, duplicated, conflicting, out-of-range, wrong-wire, and
out-of-grammar pointers; and mismatched reduced bytes. The gateway suite additionally
covers: header emitted for an applied reduction and for a partial-content reduction, header
absent in shadow mode / kill switch / unsupported paths / every fail-open path, header never
forwarded upstream and never written into receipts, and a manifest that cannot be built
failing open without a header.
