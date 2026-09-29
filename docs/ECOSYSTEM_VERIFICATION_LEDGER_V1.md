# Ecosystem verification ledger V1

Snapshot: **2026-09-20T00:18:05Z** (UTC).  This is a read-only refresh of public
repository metadata for T15/E2.  It does not install code, weights, or plugins; it does
not upload project data; and it does not change any adoption, training, provider, or
deployment gate.

## Method and evidence levels

- **Source audit** means the repository page and its public GitHub REST metadata were
  read at the snapshot time.  `stars` is the value returned by
  `GET https://api.github.com/repos/{owner}/{repo}`; it is a social signal, not a
  quality, safety, accuracy, or adoption measure.
- **Author claim** means a benchmark or capability statement in the project's own
  README/docs.  It is not a local reproduction.
- **Local reproduction** and **paired comparison** are deliberately not asserted by
  this ledger; those require separate frozen protocols and receipts.
- Comparison scope for every star count is **same-time public GitHub repository
  snapshot only**.  Counts are not normalized for age, visibility, forks, or activity,
  and must not be compared with model quality or financial performance.

## Verified repository records

| Project | Stars | Evidence level | Primary source (repo) | Metadata source | Snapshot (UTC) | Comparison scope |
|---|---:|---|---|---|---|---|
| [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) | 4,217 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/tamaratran/fast-jev-compaction) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [jev-visual](https://github.com/hr98w/jev-visual) | 145 | source audit | repo README/docs | [GitHub API](https://api.github.com/repos/hr98w/jev-visual) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [laya](https://github.com/NandhaKishorM/laya) | 1,263 | source audit | repo README/release | [GitHub API](https://api.github.com/repos/NandhaKishorM/laya) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [reflex](https://github.com/kshetrajna12/reflex) | 75 | source audit | repo README/docs | [GitHub API](https://api.github.com/repos/kshetrajna12/reflex) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [json-render](https://github.com/vercel-labs/json-render) | 16,851 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/vercel-labs/json-render) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [fx](https://github.com/vercel-labs/fx) | 3,071 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/vercel-labs/fx) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [ai-python](https://github.com/vercel-labs/ai-python) | 184 | source audit | repo README/docs | [GitHub API](https://api.github.com/repos/vercel-labs/ai-python) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [cline/plugins](https://github.com/cline/plugins) | 27 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/cline/plugins) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [kev](https://github.com/jaredpalmer/kev) | 516 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/jaredpalmer/kev) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [nimble](https://github.com/bespokelabsai/nimble) | 428 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/bespokelabsai/nimble) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [jevinci](https://github.com/achimala/jev-paint) | 36 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/achimala/jevinci) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [postgres-Jev](https://github.com/luiginotmario/postgres-Jev) | 0 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/luiginotmario/postgres-Jev) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [jev_stock](https://github.com/sosopop/jev_stock) | 7 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/sosopop/jev_stock) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) | 16 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/rorshopping/jev-on-a-laptop) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | 1,929 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/TheoLeeCJ/SemIf) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [JEV-CPU](https://github.com/leesk212/JEV-CPU) | 1 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/leesk212/JEV-CPU) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |
| [NanoJev upstream](https://github.com/TianyuCodings/NanoJev) | 870 | source audit | repo README/tree | [GitHub API](https://api.github.com/repos/TianyuCodings/NanoJev) | 2026-09-20T00:18:05Z | same-time public GitHub snapshot only |

## Targeted source addendum — 2026-09-21

This addendum is a later targeted audit and is not part of the same-time star-count comparison above.

| Project | Snapshot facts | Evidence level | Adoption relevance | Boundary |
|---|---|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | revision `ca3ba65f…`; 2,138 stars at API read; MIT code; Apple MLX, direct-logit, shared-prefix and parallel-suffix paths | source audit + author claims; no local reproduction | reference implementation for direct-logit readout, prefix reuse, evidence verification, and option-order testing | preserve MIT notice; upstream model/data terms remain separate; referenced TypeSafe records are evaluation-only |
| [JevBench](https://github.com/fstandhartinger/jevbench) / [Benchmark Heaven](https://benchmarkheaven.com/jev-models) | revision `83831807…` (`v1.2.4`); 11 stars at API read; MIT harness/public tasks; read-only results API | source audit + external benchmark protocol; no NanoJev run yet | replaces a bespoke community-wide leaderboard; provides frozen typed-decision tasks, four raw axes, adapters, and maintainer-held-out evaluation | public tasks are evaluation-only; do not train or calibrate on them; do not vendor other systems' outputs or official Jev responses |
| [laya-mlx](https://github.com/mizorewww/laya-mlx) | revision `fc1df628…`; 856 stars at API read; Apache-2.0 code with `NOTICE`; independent Apple-Silicon MLX port of Laya; packaged Python API and Snake application | source audit + author claims; no NanoJev local reproduction or paired comparison | application/runtime reference for a local typed-decision package, batched Choice/Score/Noul API, visible deterministic safety layer, opt-in compile/cache controls, numerical-parity checks, memory-stability checks, checksummed weights, and reproducible performance receipts | independent port, not an official upstream release; retain Apache-2.0 license and `NOTICE` if code is reused; model weights and upstream artifacts have separate provenance/licenses; M3 Max latency and Snake outcomes do not transfer to NanoJev; do not import its outputs or benchmark rows into training/calibration |

The repository pages are the primary sources for existence, README claims, and linked
code/docs.  The API responses are the primary source for the numeric star field only.
No row above is evidence that a project is safe to deploy, calibrated, financially
useful, faster on this Mac, or compatible with NanoJev's contracts.

## Claims intentionally left unverified

The following claims remain **unverified** and are excluded from the numeric table and
from adoption decisions: Atomic, Jevinik, the Monad live-trading bot, the DuckDB
extension, the three cost case studies, the 2,276-like objection thread, Decider-2B,
System-One 4B, the Jev-compatible public API, and the “Jev-ify any HF model” library.
No matching primary source was located in this refresh.  Their comparison scope is
**none**; they must not be quoted as facts until a stable primary URL, snapshot time,
and evidence-level record is added here.

## Impact on the roadmap

The original refresh closes metadata bookkeeping for one T15 cycle. The targeted addendum changes
future task priority, not current capability: use SemIf as an architecture/test reference,
laya-mlx as a local application/runtime evidence reference, and JevBench as the external benchmark
path instead of extending a bespoke community leaderboard.
It does not alter the T8g provenance gate, training isolation, active-pruning gate, financial gates,
or deployment status. The `achimala/jevinci` API request redirected to the canonical
`achimala/jev-paint` repository; that redirect is not an identity or capability claim. Any future
numeric refresh requires a new timestamp and explicit evidence level.
