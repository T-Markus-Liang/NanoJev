# Venue data licensing audit V1

Written because the earlier claim "Bybit/Aster/Hyperliquid terms are unverified" was too coarse:
after the egress proxy was enabled, **some of those pages became readable and one of them
contains a clause that directly describes what this project does.**

This page records only what was actually read, with the exact source, and separates read facts
from inference. Nothing here is legal advice.

## What was read, and how

| Venue | Source | Readable here? | Method |
|---|---|---|---|
| **Binance Vision** | `data.binance.vision/Binance_Vision-Terms_of_Use.pdf` | ✅ read in full | direct HTTPS (read during the P1 work package) |
| **Aster** | `docs.asterdex.com/resources/terms-and-conditions` — "Terms & Conditions", **Last updated February 25, 2026** | ✅ read in full | egress proxy (727,616 B; 20,394 chars of text) |
| **Bybit** | `www.bybit.com/en/legal/terms-of-service` | ❌ **not readable** | see below |
| **Hyperliquid** | `hyperliquid.xyz/terms`, `app.hyperliquid.xyz/terms` | ❌ **not readable** | see below |

Enabling the local egress proxy was decisive: `docs.asterdex.com` returned HTTP 200 through it
and HTTP 000 direct. This is the same proxy that unblocked the Aster and Bybit **market data**
endpoints.

## Binance Vision — read, and it constrains the project

- Licence: **CC BY-NC-SA 4.0 — non-commercial only**.
- §4.1 permits **research use**.
- **§4.2 prohibits** "live proprietary trading execution, automated commercial order
  generation, or signal distribution to third parties for direct or indirect compensation."
- §3.4 excludes commercial licensing entirely.

**Consequence:** offline research and paper backtesting are inside the permission; a **live
execution** phase is not. If this archive becomes the only price series, reaching an execution
phase later forces re-deriving features and labels on a different series — earlier results do
not transfer.

## Aster — read, and §6.2(e) describes this project

The Terms are explicit that they govern API access:

> **§1.1** — "…governing your access to and use of the Aster platform, including our website
> …, decentralized applications, **APIs**, and any related services (collectively, the
> 'Services')."

Two clauses bear directly on automated market-data fetching:

> **§6.1(b)** — "…perform, republish, **download**, store, or transmit any of the material on
> our platform **without our prior written consent**."

> **§6.2(e)** — "To use any **automated means, including bots, scrapers, or spiders**, to
> access or use the Services **without our express permission**."

Keyword scan of the full text: `research` **0** occurrences, `commercial` **0**,
`market data` **0**, `algorithmic` **0**. **There is no research or non-commercial carve-out
in these Terms.**

The Aster API documentation (`for-developers/aster-api`) grants nothing either — it describes
creating an API key with configurable read/write permissions, i.e. **authenticated** use. No
separate API terms page exists in the documentation index (`llms.txt` lists only the one
Terms & Conditions page).

### Verdict for Aster — genuinely ambiguous, both readings are defensible

- **Literal reading:** §6.1(b) requires prior written consent to download their material and
  §6.2(e) requires express permission for automated access. The public market-data endpoints
  this project calls are automated access, and no permission has been obtained. Under this
  reading **the fetch already performed is outside the permission.**
- **Reasonable counter-reading:** Aster publishes these exact endpoints as its documented
  public API for programmatic consumption. Using a published API as documented is ordinarily
  the intended use, and §6.2(e) is aimed at scraping the website rather than at calling the
  API the venue itself documents.
- **What resolves it:** either an API-specific terms document (none exists that I could find)
  or a written statement from Aster. It cannot be resolved by reading more pages.

## Bybit — could not be read, for a mechanical reason

Bybit's legal pages are a JavaScript single-page application. All of these returned a shell
with no terms text:

| Attempt | Result |
|---|---|
| `curl` direct, 7 paths | HTTP 000 |
| `curl` via proxy, browser User-Agent, `-L` | HTTP **200**, but 25,371 B of SPA shell — extracted text was **16 characters** |
| `www.bybit.com/api/legal/terms-of-service` | HTTP 200, 89,841 B, but it is the site's **i18n string table** (marketing copy), not the terms |
| Headless browser | `net::ERR_HTTP2_PROTOCOL_ERROR` — **the browser does not use the egress proxy**, and Bybit's direct route is broken in this environment |

So Bybit's actual terms text remains **unread**, and that is an environment limitation, not a
judgement about the content.

## Hyperliquid — no terms page exists in the public docs

| Attempt | Result |
|---|---|
| `hyperliquid.xyz/terms` | HTTP **403** |
| `app.hyperliquid.xyz/terms` | HTTP 200 but a **JavaScript shim**, not terms text |
| `hyperliquid.gitbook.io/hyperliquid-docs` | HTTP 200; navigation contains **no legal/terms section** |
| `…/llms.txt` (full docs index) | grep for `term|legal|licen|disclaim|commercial` → **no matches** |

Hyperliquid's public documentation does not include terms of use. Whether a binding document
exists elsewhere (for example in the on-chain governance record or the app itself) was not
established here.

## What the project owner's authorisation does and does not do

The owner has authorised using these data sources. **That settles the project's decision**:
the work proceeds, and this document does not reopen it.

What it cannot do is change what a venue's own clause says. Where a clause requires the
*venue's* permission — as Aster's §6.2(e) does — the project owner's authorisation is a
different thing from the venue's permission, and the gap is real regardless of who signs off
internally. That is the entire content of the earlier remark; it was not a request to
re-confirm anything.

**Practical consequences, in order of how soon they bite:**

1. **Paper/research use continues.** Nothing here blocks the offline backtesting this project
   is doing, and the Binance research permission covers the primary source.
2. **Aster carries the only concrete, read, unresolved prohibition.** Options are: request
   written permission from Aster; obtain and use an authenticated API key under whatever terms
   accompany it; stop using Aster data; or record the risk as accepted. Only the first two
   would close the gap rather than leave it open.
3. **Bybit and Hyperliquid are unknowns, not clearances.** No clause was read, so nothing is
   established in either direction.
4. **Live execution is a separate question from data rights** and is blocked for Binance by
   §4.2 independently of everything above.

## Reproduction

```bash
P=http://127.0.0.1:7890
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
curl -s --proxy "$P" -L https://docs.asterdex.com/resources/terms-and-conditions -o /tmp/aster_terms.html
curl -s --proxy "$P" -A "$UA" -L https://www.bybit.com/en/legal/terms-of-service -o /tmp/bybit_terms.html
curl -s --proxy "$P" -L https://docs.asterdex.com/llms.txt
```
