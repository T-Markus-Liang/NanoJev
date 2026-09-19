# Financial point-in-time data plan V1 — crypto PERPETUAL CONTRACT trading only

Status: **DRAFT — pending human review at gate R1. Not approved. No data has been purchased, downloaded, or connected to any broker or venue.**

**Revision 2 (2026-09-19): R1-level scope change applied by user decision.** The financial track covers **crypto secondary-market PERPETUAL CONTRACT trading ONLY, never spot**. The venue set is exactly **Binance, Bybit, Aster, Hyperliquid**. The spot-era long-only assumption is removed. Revision 1's recommendation of Binance Vision *spot* data is withdrawn.

Work package: [P1 of the handoff plan](CURRENT_PROGRESS_AND_HANDOFF.md). Read with [Track B of the V2 roadmap](NANOJEV_V2_ROADMAP.md) (B1 state contract, B2 calibrated targets, B3 baselines, B5 validation protocol) and the executable contract in [Financial PIT V1](FINANCIAL_PIT_V1.md) / [`scripts/financial_pit_v1.py`](../scripts/financial_pit_v1.py).

Companion machine-readable deliverables:

- [`research/financial_source_manifest_v1.json`](../research/financial_source_manifest_v1.json) — per-venue access method, licence, terms URL, permission verdicts, perp field inventory, timestamp semantics, revision policy, delisting coverage and decision record.
- [`research/financial_experiment_protocol_v1.json`](../research/financial_experiment_protocol_v1.json) — draft event/label/feature/universe/split/holdout/seed contract plus the perp semantics section and the two-sided action space.

This document surveys candidate venues only. It contains **no profitability claim, no strategy recommendation, and no live-trading proposal**, and it does not authorize order submission, broker credentials, account creation, API keys, or paid data acquisition.

---

## 0. Three load-bearing warnings

### 0(a) The previously recommended Binance archive is non-commercial and forbids live execution

**This is the single most consequential finding in this revision and it is stated first so it cannot be buried.**

The Binance Data Collection / Binance Vision public archive — the source revision 1 recommended — is licensed **CC BY-NC-SA 4.0, non-commercial only**, and its **§4.2 expressly prohibits live proprietary trading execution**. The terms document is `BINANCE VISION DATASET TERMS`, Version 1.0, last updated **August 26 2026**, read in full from <https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf> (4 pages). The operative clauses, quoted verbatim:

| Clause | Text |
|---|---|
| 3.1 | "Datasets are provided to You under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (\"CC BY-NC-SA 4.0\")…" |
| 3.4 | "Commercial licensing is strictly excluded under this agreement; any commercial utilization requires a separate, written enterprise data license agreement executed with Binance." |
| 4.1 | "Permitted non-commercial uses are limited to academic research, non-monetized open educational projects, algorithmic historical backtesting for purely personal non-production research, and non-commercial open-source data science evaluations." |
| 4.2 | **"You shall not use the Datasets for live proprietary trading execution, automated commercial order generation, or signal distribution to third parties for direct or indirect compensation."** |
| 4.3 | "You shall not license, sub-license, sell, rent, lease, host, or commercially exploit the Datasets or any derivative real-time or historical data feeds." |
| 4.4 | "You shall not integrate the Datasets into commercial financial products, hedge fund management tools, paid subscription newsletters, or commercial trading bot platforms." |
| 4.5 | Redistribution of derivative works requires attribution and identical CC BY-NC-SA 4.0 terms. |
| 7.2 | "You shall not deploy scraping routines, scripts, or continuous automated queries designed to circumvent rate restrictions or impair platform hosting operations." |

**What this means for a perpetual-contract trading program that may later want execution.**

1. **Offline research is permitted. Live execution is not.** §4.1 permits "algorithmic historical backtesting for purely personal non-production research". That covers exactly what P1/P2/P3 are. §4.2 then removes the execution phase entirely.
2. **§4.2 is unconditional.** It is not limited to redistribution, to compensation, to commercial accounts, or to trading style. There is no size threshold and no revenue threshold in the text.
3. **The prohibition covers automated order generation directed at the market**, i.e. precisely what an automated perp strategy does. §4.4 independently bars integrating the data into "commercial financial products, hedge fund management tools, … commercial trading bot platforms", which describes a for-profit trading operation.
4. **Copyleft propagates.** §4.5 keeps any redistributed derivative work under CC BY-NC-SA 4.0, so a published model, feature set, or dataset derived from this archive inherits the non-commercial restriction.
5. **The strategic cost is a data-rights migration.** If this archive becomes the only price series behind features and labels, then reaching an execution phase later means re-deriving those features and labels on a different series. Mark price, bar boundaries, print selection, oracle construction, and funding timestamps all differ between an archive and a live or licensed feed, so **earlier results do not transfer**; they are invalidated. That cost is paid either now, by choosing a licensable source, or later, by redoing the work.
6. **This is a dataset restriction, not necessarily a venue restriction.** The clauses govern the *Datasets* distributed through `data.binance.vision`. Binance's live market-data and trading APIs are governed by separate Binance terms, and `www.binance.com`, `developers.binance.com` and `fapi.binance.com` were all **unreachable** from the review environment, so whether the live API carries different permissions is an **open question (Q2)**, not a settled yes.

**Alternatives, ranked.**

| Rank | Alternative | Cost | What it buys | What it costs you |
|---|---|---|---|---|
| 1 | Research on the Binance archive now; accept that execution requires new data rights later | USD 0 today | Fastest route to pipeline validation | A deferred migration; results tied to an archive price series may not transfer |
| 2 | Use each venue's live REST/WS market API for research, archive only as cross-check | USD 0 *if* the venue terms allow it | Avoids building on the non-commercial dataset | Bybit/Aster/Hyperliquid terms unread; Hyperliquid's API caps candles at 5,000, far too short for the folds |
| 3 | Written enterprise data licence from Binance (§3.4 contemplates exactly this) | `unverified: quote_on_request` | Makes the archive itself execution-compatible | Sales cycle; contract; out of scope under the current no-purchase constraint |
| 4 | A licensed third-party redistributor for the same venues | `unverified: quote_on_request` | Decouples data rights from the venue | Institutional pricing; sales cycle |
| 5 | Change the venue set to one whose terms clearly permit the intended use | unknown | Possibly removes the conflict | Cannot be evaluated until the other three venues' terms are readable |

**The decision R1 must actually make:** does this project intend to ever submit live orders? If yes, the Binance archive cannot be the sole price source, and that must be decided now rather than after a model exists. The plan's position is that offline-only research is the correct scope for this stage, but that position must be an explicit R1 ruling, not a default that quietly forecloses execution.

### 0(b) Setting an availability timestamp does not prove point-in-time authenticity

Every source here is consumed as a *current snapshot* of history. Writing `available_ns` into a record proves only that our own code declared a time; it cannot prove that the byte read today is the byte that existed then. The validator in `scripts/financial_pit_v1.py` — per its own documentation — "can detect inconsistent timestamps, not dishonest or incorrectly reconstructed timestamps."

This warning is **stronger for Binance than revision 1 assumed**, because the archive's own README states: *"Archived files may be updated at a later date as a result of recently discovered issues"*, and publishes an updates table recording a 2022-08-08 kline update and a 2022-04-21 aggregate-trade update, each with CHECKSUMs of the replaced and replacement files. That is direct evidence that **the archive has been rewritten in place**. Point-in-time authenticity requires either a vendor contract guaranteeing as-of vintages or an independently hash-pinned archive observed over time. We have neither. This is an **open question (Q3)**, not a solved problem, for every venue.

### 0(c) Source licensing must be independently confirmed before use — three of four venues' terms could not be read at all

The licence notes below were read from publicly published terms at the cited URLs, on the review date, by an automated agent. Reading a public web page is **not** a legal opinion and **not** confirmation that a licence covers this project's intended use.

**Critical honesty point for this revision: only Binance's terms could be read.** Bybit's, Aster's and Hyperliquid's terms pages were **not reachable from this environment at all**. They are recorded as `unverified` with the exact URL and the exact failure mode. **No terms are invented, summarised from a search result, or inferred from the existence of a public API.** The existence of a documented API is not a licence to use it for a particular purpose.

---

## 1. Review scope and evidence standard

Reviewed 2026-09-19. **Four venues** were surveyed: Binance, Bybit, Aster, Hyperliquid. Equities, CME futures, FX, macro and the second crypto venue from revision 1 are out of scope for this track.

Evidence rules applied:

- Every factual claim carries the URL it was read from.
- Where a term, price, or field could not be retrieved from a primary page, the entry says `unverified` and the gap is listed in §9. Nothing is inferred from a reseller summary, a blog, a forum post, or a prior model's memory.
- Where a venue's own documentation states a limitation (Binance's non-commercial clause, Bybit's funding-history pagination trap, Aster's deposit gate, Hyperliquid's 5,000-candle cap), that limitation is quoted rather than paraphrased away.
- **Reachability is recorded explicitly**, because it determines what could be established at all. The review environment reached `data.binance.vision`, `s3-ap-northeast-1.amazonaws.com`, `raw.githubusercontent.com`, `api.github.com`, `hyperliquid.gitbook.io` and `api.hyperliquid.xyz`. It did **not** reach `www.bybit.com`, `api.bybit.com`, `docs.asterdex.com`, `www.asterdex.com`, `fapi.asterdex.com`, `www.binance.com`, `developers.binance.com`, `app.hyperliquid.xyz`, or `hyperliquid.xyz/terms` (HTTP 403). Reachability from a sandbox says nothing about a venue; it says everything about which permission questions remain open.

**Scope reduction note.** Revision 1's per-source records for the ten non-perpetual sources were **removed** rather than retained as stale alternatives, so that no reader mistakes an equity or CME option for a live candidate. Revision 1's text remains in this file's repository history.

---

## 2. Survey summary

| Venue | Access method | Terms URL | Terms reachable? | Research use | Automated/algorithmic trading | Live-execution clause | Perp data depth |
|---|---|---|---|---|---|---|---|
| **Binance** | Public bulk archive (HTTPS/S3-style, no key) + live REST/WS API | <https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf> | **yes, read in full** | **permitted, NON-COMMERCIAL ONLY** (§4.1) | **prohibited for commercial order generation** (§4.2, §4.4) | **§4.2 prohibits "live proprietary trading execution"** | Strong for price, funding, OI, positioning; **no liquidation/ADL history** |
| **Bybit** | Public REST v5 + WebSocket; no bulk archive found | <https://www.bybit.com/en/legal/terms-of-service> | **no — host unreachable** | `unverified` | `unverified` | `unverified` — not read | Funding, OI (multi-grid), mark/index/premium klines, contract specs; **no liquidation history** |
| **Aster** | Public REST v3 + WebSocket at `fapi.asterdex.com` (Binance-shaped) | <https://docs.asterdex.com/resources/terms-and-conditions> | **no — host unreachable** | `unverified` | `unverified` | `unverified` — not read | Funding (+interval config), mark/index/premium, contract specs; **no open interest**; liquidation only via deposit-gated auth or 1 s stream |
| **Hyperliquid** | On-chain: `POST /info` + public S3 archive `s3://hyperliquid-archive` | <https://hyperliquid.xyz/terms> | **no — HTTP 403 (S3 AccessDenied)** | `unverified` | `unverified` | `unverified` — not read | Best contract-spec/state object (meta, marginTiers, isDelisted); hourly funding; **candles capped at 5,000**; archive has no candles |

Count: **4 venues surveyed. 1 licence verified (Binance). 3 unverified (Bybit, Aster, Hyperliquid).**

---

## 3. Per-venue detail

### 3.1 Binance — `binance_perp_bulk_archive` — **recommended first market**

**Access method.** Public HTTPS object archive, no account and no key. Web index at <https://data.binance.vision/>; underlying bucket `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision`. Every archive has a sibling `.CHECKSUM` file. Verified by unauthenticated `HEAD` requests during this review. **Perp-only path grammar** (all probed):

```text
data/futures/um/{monthly|daily}/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM[-DD]}.zip
data/futures/um/{monthly|daily}/markPriceKlines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM[-DD]}.zip
data/futures/um/{monthly|daily}/indexPriceKlines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM[-DD]}.zip
data/futures/um/{monthly|daily}/premiumIndexKlines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM[-DD]}.zip
data/futures/um/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{YYYY-MM}.zip
data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip
data/futures/um/{monthly|daily}/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM[-DD]}.zip
data/futures/um/{monthly|daily}/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM[-DD]}.zip
data/futures/um/daily/bookDepth/{SYMBOL}/{SYMBOL}-bookDepth-{YYYY-MM-DD}.zip
data/futures/cm/...   (COIN-M inverse: same families, with the coverage exceptions in §3.1.4)
```

**Licence: CC BY-NC-SA 4.0, non-commercial, and §4.2 prohibits live execution.** See §0(a) for the full clause table and the alternatives. Research use is permitted under §4.1 for non-commercial personal non-production backtesting; automated commercial order generation is prohibited by §4.2 and §4.4. **No live-execution right exists under these terms.**

**Verified perp fields** (file headers and rows actually read; six files under 30 KB each, written only to `/tmp`):

| Dataset | Header read | Verified facts |
|---|---|---|
| `fundingRate` | `calc_time,funding_interval_hours,last_funding_rate` | 93 funding-settlement rows in the `2023-01` file; first row `1672531200000,8,0.00010000`; `2020-01` first row `1577836800000,8,-0.00012359`. Probes from `2020-01` to `2026-08` returned 200; `2019-09`/`2019-12` returned 404. **The funding interval is carried on every row** — no interval join needed. |
| `klines` (last price) | `open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore` | Last **traded** price, not mark price. |
| `markPriceKlines` | same column family | 745 rows for a 1 h month. **This is the perp label price series.** |
| `indexPriceKlines` | same column family | 745 rows for a 1 h month. UM verified; **CM `indexPriceKlines` returned 404 at the probed path → unverified**. |
| `premiumIndexKlines` | same column family | Signed **decimal**, e.g. `-0.00022221` — not basis points. UM and CM both 200. |
| `metrics` (daily) | `create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio` | **288 rows/day at exactly 5-minute spacing.** `create_time` is a **UTC datetime string** (`2023-01-02 00:00:00`), not an integer epoch. CM version has the three positioning columns **empty**. Monthly metrics path returned 404. |

**Not available from Binance's archive.** **Liquidation/ADL history**: all probed liquidation paths returned 404 (`um/daily/liquidationSnapshot`, `um/monthly/liquidationSnapshot`, `um/daily/forceOrders`, `um/monthly/liquidation`, `cm/daily/liquidationSnapshot`). Binance documents forced liquidations as a WebSocket `forceOrder` stream, which is not historically reconstructible. Also 404: `bookTicker` (daily and monthly). **Leverage and margin mode** are account state, not public market data, and belong to the P2 simulator.

**Timestamp semantics.** Millisecond integer epoch in every futures sample read (13-digit). The README's microsecond note names **SPOT only** — *"The timestamp for SPOT Data from January 1st 2025 onwards will be in microseconds"* — so the known unit break is out of scope for a futures-only track, though the futures unit must still be **asserted** at ingestion rather than assumed, because this review did not fetch a post-2025 futures file. **The metrics exception matters**: one loader assuming integer milliseconds for every archive file will silently corrupt the open-interest series.

**Revision policy.** No as-of vintages. §5.3 reserves the right to modify, redact or discontinue access to any dataset at any time. The README documents that archived files **have been replaced** for quality reasons, with a linked update table. That is measured evidence of in-place rewriting.

### 3.2 Bybit — `bybit_v5_market_api` — **co-primary, licence unverified**

**Access method.** Public REST v5 and WebSocket. **No bulk historical archive was found** in the venue's documentation or its GitHub organisation. Historical depth is bounded by per-endpoint pagination and by each symbol's launch time. Documentation read from the venue's own docs source at <https://github.com/bybit-exchange/docs> (pinned commit `75994fda16e052aaad6e3fade82fd1f6fd90288e`).

**Licence: `unverified`.** Every attempted path failed with HTTP 000 (no connection): `/en/legal/terms-of-service`, `/en/legal/terms-of-service/`, `/terms-of-service`, `/en/legal`, `/en/legal/api-terms`, `/en/help-center`, `/api/legal/terms`. A hosted web fetch failed with a hostname-resolution error. **No terms text was read. Nothing is asserted about Bybit's permitted or prohibited uses.**

**Research use / algorithmic trading / live execution: all `unverified`.** The API's existence is not treated as permission.

**What a reviewer must read** (recorded in the manifest): the ToS section governing API and automated/programmatic trading; any separate API or market-data terms including retention and redistribution; any restriction on automated order placement or latency-sensitive strategies; and whether KYC/account approval is required before API market data is served.

**Perp fields (from documentation, not exercised):**

- **Funding** — `GET /v5/market/funding/history`; per-symbol interval, **not** in the response. Documentation directs the caller to `instruments-info`, which carries `fundingInterval` **in minutes** (the official USDT-perpetual sample shows `fundingInterval: 480`, i.e. 8 h). **Pagination trap, quoted:** "Passing only `startTime` returns an error. Passing only `endTime` returns 200 records up till `endTime`. Passing neither returns 200 records up till the current time."
- **Open interest** — `GET /v5/market/open-interest` at `5min/15min/30min/1h/4h/1d`, with **both a both-sides aggregate (`openInterest`) and a one-sided value (`singleOpenInterest`)**. Unit rule quoted: linear is in **base coin** (BTC), inverse is in **USD**. Upper bound is the symbol's launch time; documented caveat: "During periods of extreme market volatility, this interface may experience increased latency or temporary delays in data delivery."
- **Three price references in one response** — `tickers` carries `lastPrice`, `indexPrice`, `markPrice`, plus `openInterest`, `singleOpenInterest`, `fundingRate`, `nextFundingTime`. Mark/index/premium klines are separate endpoints; **premium-index kline is linear-only**.
- **Contract specs** — `instruments-info`: `contractType` (e.g. `LinearPerpetual`), `priceFilter.tickSize`, `lotSizeFilter.minNotionalValue`, `minOrderQty`, `maxOrderQty`, `qtyStep`, `fundingInterval`, and `deliveryTime`, documented as covering **"Perpetual delisting time"** — a genuine point-in-time delisting signal.
- **Unit swap trap** — kline `volume` is base coin for linear and **quote coin (USD)** for inverse, with `turnover` the mirror image. Comparing across contract types without conversion is wrong.
- **ADL / insurance** — `GET /v5/market/adl-alert` (updated every 1 minute) returns current `balance`, `pnlRatio`, `insurancePnlRatio`, `adlTriggerThreshold`, `adlStopRatio`, and a deprecated always-empty `maxBalance`; `GET /v5/market/insurance` gives a **24-hour** pool balance. **This is current state, not event history.** No queryable historical ADL or liquidation log was found.
- **Risk limits** — `GET /v5/market/risk-limit` publishes leverage tiers.

### 3.3 Aster — `aster_futures_api` — **co-primary, licence unverified, no open interest**

**Access method.** Public REST v3 + WebSocket at base endpoint **`https://fapi.asterdex.com`**, documented as Binance-Futures-shaped. Public market-data endpoints require no signature. Documentation read directly from <https://github.com/asterdex/api-docs>.

**Licence: `unverified`.** `docs.asterdex.com/resources/terms-and-conditions`, `www.asterdex.com/en/terms-of-service` and `docs.asterdex.com` all failed to connect (HTTP 000). A search result identified the canonical terms location, but the page was never served. **No terms text was read.**

**Deposit gate (read from the API README, and it matters):**

> "Starting from **September 1, 2026, 00:00 UTC**, all authenticated Spot V3 / Futures V3 endpoints (account, order, trade, and other USER_DATA / TRADE endpoints) are only accessible after the main wallet linked to the Aster account has completed a deposit. **Agent Wallet and Builder endpoints, and public market data endpoints, are unaffected.** Until the deposit is completed, affected requests return `{"code":-5050,...}`."

So public historical research is **not** gated, but the only REST route that returns liquidation and ADL orders is a `USER_DATA` endpoint behind a **wallet deposit** — an account action that is out of scope. The README also notes that from **March 25 2026** new V1 API keys can no longer be created (existing keys keep working), so any future integration must target V3.

**Perp fields (from documentation):**

- `GET /fapi/v3/klines`, `/markPriceKlines`, `/indexPriceKlines`.
- `GET /fapi/v3/premiumIndex` returns **`markPrice`, `indexPrice` and `lastFundingRate` together**.
- `GET /fapi/v3/fundingRate` → `symbol, fundingRate, fundingTime`; **the interval is not here**.
- `GET /fapi/v3/fundingRateConfig` → `interestRate, time, fundingIntervalHours, fundingFeeCap, fundingFeeFloor`. The documented sample shows **two symbols with different intervals (8 h and 4 h)**, so the interval is per-symbol **and changes over time**. Whether this endpoint serves history or only current config is **unverified**.
- `GET /fapi/v3/exchangeInfo` → `tickSize`, `stepSize`, `minQty`, `liquidationFee`, `MIN_NOTIONAL`. **Documented trap, quoted:** `pricePrecision` is annotated *"please do not use it as tickSize"* and `quantityPrecision` *"please do not use it as stepSize"*.
- `GET /fapi/v3/forceOrders` accepts `autoCloseType=LIQUIDATION|ADL`; order statuses include `CALCULATED - Liquidation Execution`, `NEW_INSURANCE`, `NEW_ADL - Counterparty Liquidation`. **This is the only venue in the set whose read documentation exposes a queryable route naming both liquidation and ADL** — but it is authenticated and deposit-gated.
- Liquidation streams `@forceOrder` / `!forceOrder@arr`: *"For each symbol, only the latest one liquidation order within 1000ms will be pushed as the snapshot."* **Lossy by construction.**
- Trades documentation: *"the insurance fund trades and ADL trades won't be returned"* — ADL activity is excluded from the trade tape too.

**Not available: open interest.** The V3 futures market-data section lists no open-interest endpoint, no open-interest history, and no long/short-ratio endpoint. That is a material gap for a perp data model.

### 3.4 Hyperliquid — `hyperliquid_onchain` — **co-primary, licence unverified, depth-capped**

**Access method.** Three public routes: (1) `POST https://api.hyperliquid.xyz/info` (reachable — a bare GET returned the expected HTTP 405 on a POST-only route); (2) a public S3 archive `s3://hyperliquid-archive`; (3) on-chain/explorer APIs. Being a blockchain, order flow is attributable to blocks — a property no CEX here exposes.

**Licence: `unverified`.** `https://hyperliquid.xyz/terms` returned **HTTP 403 with an S3 `AccessDenied` body**; `app.hyperliquid.xyz` did not connect; the documentation site exposes no terms/legal page; the foundation domain returned 404 for `/terms-of-use` and `/legal/terms`. **No terms text was read.** A secondary source (an SEC filing quoting third-party terms) suggests the interface's terms define a **"Restricted Persons"** category, but that is a quotation inside another document and was not read from a primary page — recorded as an unverified signal, not a finding.

**Perp fields (documentation read):**

- `meta` → universe (`name`, `szDecimals`, `maxLeverage`, `onlyIsolated`, **`isDelisted`**, `marginMode` where `strictIsolated`/`noCross` are documented) plus `marginTables` with `marginTiers[] {lowerBound, maxLeverage}`. **The richest contract-specification response in the set**, and the only one with a delisting flag.
- `metaAndAssetCtxs` → per instrument: `funding`, `openInterest`, `markPx`, `midPx`, `oraclePx`, `premium`, `prevDayPx`, `dayNtlVlm`, `impactPxs`. **One call yields the complete perp state vector.** Caveat: the universe and context arrays are **positionally aligned, not keyed**, and HIP-3 coins carry a dex prefix (`xyz:XYZ100`) — an off-by-one join mislabels instruments.
- `fundingHistory` → `coin, fundingRate, premium, time`. **Funding is paid hourly** (documented: the 8-hour formula is applied at one eighth per hour; "The premium is sampled every 5 seconds and averaged over the hour"). Finer-grained than any 8-hour venue, so cross-venue funding pooling needs time-grid alignment.
- `predictedFundings` → cross-venue predicted funding rates (first perp dex only).
- `candleSnapshot` → intervals `1m…1M`, with the hard limit quoted: **"Only the most recent 5000 candles are available."** 5,000 daily candles ≈ 13.7 years (fine for the folds); 5,000 hourly ≈ 7 months; 5,000 1-minute ≈ 3.5 days.
- **Historical archive** `s3://hyperliquid-archive`: `market_data/[date]/[hour]/[datatype]/[coin].lz4` (L2 book snapshots) and `asset_ctxs/[date].csv.lz4` (asset contexts, i.e. funding/OI/mark/oracle). Quoted caveats: uploaded *"approximately once a month. There is no guarantee of timely updates and data may be missing"*, **"No other historical data sets are provided via S3 (e.g. candles or spot asset data)"**, and **"the requester of the data must pay for transfer costs."** Trade and event history live in separate buckets (`node_fills_by_block`, `explorer_blocks`, `replica_cmds`, `misc_events_by_block`).
- **Tick/lot size** — derived, not stored: prices carry at most **5 significant figures** and no more than `MAX_DECIMALS - szDecimals` decimals, where `MAX_DECIMALS` is 6 for perps and 8 for spot; integer prices are always allowed. Minimum notional appears as an order-time rejection status (`minTradeNtlRejected`).
- **Liquidations** — documented as book-first market orders, with a **liquidator-vault backstop** below 2/3 of maintenance margin (HLP), 20 % partial liquidation above 100 k USDC followed by a 30-second full-size cooldown, and no clearance fee. **There is no ADL quantile endpoint and no public liquidation-event dataset**; the mechanism is a vault, not a CEX-style ADL queue, so cross-venue ADL features are not comparable.
- **Rate limits** — 1,200 aggregated weight/minute per IP; info requests weight 20, `l2Book`/`allMids`/`clearinghouseState` weight 2, `userRole` weight 60, with per-20-item and per-60-item added weights on `fundingHistory` and `candleSnapshot`. Documented guidance: **"For large batch requests, use the S3 bucket instead."**

---

## 4. Recommendation: minimal first market

### 4.1 Recommendation

**Crypto PERPETUAL CONTRACT daily bars, mark-price labelled, USDT-margined linear instruments first, with the Binance Bulk archive as the primary price source and Bybit, Aster and Hyperliquid as coverage and holdout sources.**

Concretely: `data/futures/um/monthly/markPriceKlines/{SYMBOL}/1d/` as the first ingestion target, joined to `data/futures/um/monthly/fundingRate/` and `data/futures/um/daily/metrics/` for funding and open interest, with `1m` and `aggTrades` reserved for a later pre-registered microstructure extension.

**Why this is the minimal defensible first market:**

1. **Licence is explicit, published, readable, and non-commercial.** CC BY-NC-SA 4.0 under a versioned terms document, read in full. This is the only one of the four venues where anything about permitted use could be established at all — which is simultaneously its strength (clarity) and its central limitation (§0(a)).
2. **Zero cost, no procurement gate, no account.** §2.3 grants free access; no key, account, contract or approval is needed. R1 stays a research-design decision rather than a spending decision.
3. **Its funding archive needs no interval join.** `funding_interval_hours` is a column on every row, so the interval is data rather than an assumption — Bybit and Aster both require a separate config/interval lookup whose historical depth is unverified.
4. **Its open-interest series is the most granular and most clearly documented in the set** — 5-minute, daily files, with an explicit both-sides aggregate, and the CM emptiness documented rather than hidden.
5. **All three price references are archived** (last, mark, index, premium), which is exactly the set a perpetual data model needs, under one licence and one path grammar.
6. **Delisted instruments remain addressable**, enabling the survivorship control the B5 requirement demands.
7. **The corporate-action class of hazard is absent**, so the pipeline can be validated without simultaneously fighting adjustment-vintage problems.

### 4.2 What the recommendation explicitly does not claim

- It does **not** claim the data is point-in-time authentic (§0b). The archive is documented as having been **rewritten in place** at least twice for quality reasons.
- It does **not** claim any execution path. **Binance's §4.2 prohibits live proprietary trading execution outright** and §3.4 excludes commercial licensing; this source cannot support an execution phase without a different data right.
- It does **not** claim perpetuals are a better research subject than anything else; the scope was fixed by user decision.
- It does **not** claim that Bybit's, Aster's or Hyperliquid's terms permit anything. All three are unread.
- It does **not** claim any performance, edge, or profitability. No model has been trained and none is authorized before R1.

### 4.3 Ranked alternatives and tradeoffs

| Rank | Option | Why | Cost | Blocking issue |
|---|---|---|---|---|
| **1 (recommended)** | Binance Bulk archive, perp datasets, 4-venue cohort | Only verified-readable licence; funding interval on every row; 5-min OI; all three price references archived; free | USD 0 | Non-commercial only; §4.2 bars execution; no liquidation history; no PIT vintages |
| 2 | Same cohort sourced from each venue's live public API | Avoids the non-commercial dataset for the unverified venues | USD 0 *if* terms allow | Bybit/Aster/Hyperliquid terms unread; Hyperliquid candles capped at 5,000; no bulk archives on Bybit/Aster |
| 3 | Enterprise data licence from Binance (§3.4) | Makes the archive execution-compatible | `unverified: quote_on_request` | Cost/timeline unknown; out of scope under the no-purchase constraint |
| 4 | Licensed third-party redistributor for the same four venues | Decouples data rights from venue relationships | `unverified: quote_on_request` | Institutional pricing; sales cycle |
| 5 | Venue set chosen for permissive terms | Removes the licence conflict at the root | unknown | Cannot be evaluated until the other three terms pages are readable |
| — | Hyperliquid-only cohort | Richest state object; on-chain determinism; delisting flag | USD 0 + S3 egress | API candles capped at 5,000; archive has **no candles** and is published "approximately once a month" with possible gaps |

**Reading the ranking.** Ranks 1 and 2 are not really different markets — they are different *data rights* for the same market. That is the real choice this revision surfaces: the price series you validate the pipeline on determines whether an execution phase is reachable at all.

---

## 5. Perp-specific data model

A perpetual contract is not a spot pair with leverage. The following observables and conventions must be represented explicitly before any feature or label is defined.

### 5.1 Funding rate, interval and settlement

Funding is a periodic peer-to-peer transfer between the long and short sides, sized to pull the perpetual toward its index reference. It is a **holding cost of a leveraged position**, not part of a gross price move.

| Venue | Funding history | Interval exposed where | Interval as read |
|---|---|---|---|
| Binance | `monthly/fundingRate` archive | **on every row** (`funding_interval_hours`) | `8` on the probed BTCUSDT rows |
| Bybit | `GET /v5/market/funding/history` | `instruments-info` → `fundingInterval` | **minutes**; official USDT-perp sample shows `480` (= 8 h) |
| Aster | `GET /fapi/v3/fundingRate` | `fundingRateConfig` → `fundingIntervalHours` | documented sample shows **8 and 4** for two different symbols |
| Hyperliquid | `POST /info` `fundingHistory` | documented in prose | **hourly**, one eighth of the 8-hour rate; premium sampled every 5 s |

**Consequences.** (i) The settlement timestamp is a discrete venue event and must be carried with its own `event_ns`, not pro-rated from bar boundaries. (ii) Because intervals differ per venue **and can change per symbol over time**, a historical funding series needs a point-in-time interval join to be interpreted, and pooling venues without aligning to each venue's own settlement grid is a unit error in the time domain. (iii) Funding belongs to the **P2 simulator**, never to the label.

### 5.2 Mark price vs index price vs last price

| Reference | What it is | Where it appears here |
|---|---|---|
| **Last price** | Most recent traded price. Reflective but noisiest; vulnerable to a single thin print. | Binance `klines`; Bybit `kline`/`lastPrice`; Aster `klines`; Hyperliquid `midPx`/`prevDayPx` |
| **Mark price** | The venue's fair-price reference used for unrealized PnL, margin and liquidations. | Binance `markPriceKlines`; Bybit `mark-price-kline`/`markPrice`; Aster `markPriceKlines`/`markPrice`; Hyperliquid `markPx` |
| **Index / oracle price** | The venue's underlying reference, built from an external basket. A spot-derived input, never a traded instrument. | Binance `indexPriceKlines` (**UM verified, CM 404**); Bybit `index-price-kline`/`indexPrice`; Aster `indexPriceKlines`/`indexPrice`; Hyperliquid `oraclePx` |
| **Premium / basis** | Signed gap between the perpetual and its index/oracle — the funding driver. | Binance `premiumIndexKlines` (signed decimal); Bybit premium-index kline (**linear only**); Aster `premiumIndex`/index-price-references; Hyperliquid `premium` |

**Protocol rule.** The V1 label is computed from **mark price at both ends**, because mark price is the price at which unrealized PnL and liquidation are actually measured — so a mark-to-mark return is the quantity a leveraged position genuinely experiences. Last-price labelling is permitted only as a separately declared robustness variant. Whenever both exist, a feature must name which one it uses in its `version` string; a silently mixed series is a defect.

### 5.3 Open interest

A **level**, not a flow, and its unit differs by contract type. Binance archives it at **5-minute** granularity in daily `metrics` files with both `sum_open_interest` and `sum_open_interest_value`. Bybit serves it at `5min/15min/30min/1h/4h/1d` and distinguishes **both-sides** (`openInterest`) from **single-side** (`singleOpenInterest`), with linear in base coin and inverse in USD. Hyperliquid carries `openInterest` in `metaAndAssetCtxs` and in the hourly `asset_ctxs` archive. **Aster exposes no open-interest endpoint at all.**

**Rule.** The unit (base coin, contracts, USD notional) and the one-sided-vs-two-sided convention must be recorded per source. A unit or convention error rescales the series by a constant factor without producing any visible error — the most dangerous class of silent bug in this model.

### 5.4 Liquidation and ADL events

| Venue | Historical liquidation events | ADL events |
|---|---|---|
| Binance | **No** — no liquidation file at any probed archive path; live `forceOrder` WS stream only | No public route read |
| Bybit | **No** — current 1-minute alert state + 24-hour insurance balance only | Partial: `adl-alert` state (`pnlRatio`, `adlTriggerThreshold`, `adlStopRatio`); no history |
| Aster | Partial: authenticated, **deposit-gated** `forceOrders`; live stream **down-sampled to one event per symbol per 1000 ms** | Partial: `autoCloseType=ADL`, statuses include `NEW_ADL`; same gate |
| Hyperliquid | **No** public event dataset; mechanism is a liquidator-vault backstop, not a CEX ADL queue | Not comparable to CEX ADL by construction |

**Rule.** Liquidation and ADL events are **features and simulator inputs, never label components**. The availability gap is first-class: the protocol's `liquidation_intensity_1d` feature is therefore **conditional** and may have to be dropped or the universe restricted. It must never be filled with zeros to keep the schema uniform, because a zero reads as "no liquidations" rather than "not observed".

### 5.5 Leverage and margin mode (isolated/cross)

Leverage is per-instrument, user-selected, and bounded by venue risk tiers that are themselves a function of position size. Hyperliquid publishes this most completely (`maxLeverage`, `marginTiers` with `lowerBound`/`maxLeverage`, and `marginMode`/`onlyIsolated`); Bybit publishes `risk-limit` tiers; Aster publishes leverage brackets in `exchangeInfo`; Binance's `exchangeInfo` was not reachable.

**Protocol rule.** For V1, **leverage and margin mode are P2 simulator inputs, not label inputs and not model targets.** A label computed at 1x notional and one computed at 20x isolated are the same gross price event; treating them differently would smuggle an execution assumption into the event definition. Liquidation and ADL consequences are explicitly out of scope for the V1 label.

### 5.6 Contract specifications

| Spec | Binance | Bybit | Aster | Hyperliquid |
|---|---|---|---|---|
| Tick size | `tickSize` (exchangeInfo — **unreachable**) | `priceFilter.tickSize` | `priceFilter.tickSize` (⚠ `pricePrecision` is **not** tick size, per the docs) | **Derived**: ≤5 significant figures and ≤ `6 - szDecimals` decimals for perps |
| Lot size | `stepSize` (**unreachable**) | `qtyStep`, `minOrderQty`, `maxOrderQty` | `stepSize`, `minQty` (⚠ `quantityPrecision` is **not** step size) | Rounded to `szDecimals` |
| Minimum notional | `minNotional` (**unreachable**) | `minNotionalValue` (sample: `5`) | `MIN_NOTIONAL` filter | Enforced at order time (`minTradeNtlRejected`) |
| Contract multiplier | `1` for linear; non-trivial for COIN-M inverse | `1` for linear; inverse is USD-denominated | not read | not applicable (on-chain, size in coin) |
| Contract type | `contractType` (filter out quarterly delivery) | `contractType` (`LinearPerpetual`) + `deliveryTime` (restates perpetual delisting) | futures only, USDT-margined | `meta` universe + HIP-3 dex prefix |

**Rule.** Contract specs are **simulator and universe-screening inputs**, required to compute a realistic minimum tradeable size in P2 and to prove an instrument was actually tradeable at a decision time. They are never label inputs.

### 5.7 USDT-margined vs coin-margined contracts

- **Linear (USDT-margined):** margin and PnL in USDT; quantity in base coin; volume/turnover quote-denominated. Binance USD-M, Bybit `linear`, Aster USDT perps.
- **Inverse (coin-margined):** margin and PnL in the **base coin**; a fixed USD notional per contract; quantity in contracts or USD. Binance COIN-M, Bybit `inverse`. **The same economic exposure produces a return series in a different numeraire.** Binance COIN-M `klines` carry `Base asset volume` and `Taker buy base asset volume` columns instead of the quote-denominated equivalents; Bybit's volume/turnover units **swap** between linear and inverse.
- **On-chain (USDC-margined):** Hyperliquid perps. A **third** numeraire base.

**Rule.** V1 declares a **single quote numeraire** and therefore a single contract-type family (USDT-margined linear). Instrument identity must encode **venue + contract type + symbol**, so a linear and an inverse contract on the same underlying are two distinct instruments and can never be silently merged. Coin-margined and USDC-margined on-chain perpetuals are in scope for the venue survey but deferred from the V1 cohort to a separately pre-registered extension.

---

## 6. Revised event, label and action contract

The full machine-readable contract is in [`research/financial_experiment_protocol_v1.json`](../research/financial_experiment_protocol_v1.json). The load-bearing changes from revision 1:

### 6.1 The spot-era long-only assumption is removed

Revision 1 declared `"direction": "long_only"`. **That is gone.** The action space is explicitly **two-sided: long / flat / short**, declared in `event_definition.two_sided_action_space`:

- The **event probability** and the **action policy** are separate objects. The model emits `P(up event)`; the policy maps that probability plus a declared cost and risk state to one of `long`/`flat`/`short`.
- **A low `P(up)` is evidence for a short position, not automatically a flat.** The long-decision and short-decision thresholds are fitted on the **calibration split only**; tuning either on test outcomes is prohibited.
- The validator accepts **exactly one event definition per cohort**, so the cohort predicts the **long-side (upward) event**; the short side is the complementary downward event over the same horizon and threshold, handled by the policy, not by a second label. A symmetric two-label cohort is prohibited by contract.
- The evaluation plan adds two required baselines that the long-only era made implicit: **always-long** and **random long/flat/short at the observed base rate with a declared seed**.

### 6.2 A gross price-move label is NOT a tradable net return

**Stated explicitly in both the event definition and the label predicate:**

> This is a **GROSS PRICE-MOVE** event label and is **NOT a tradable net return**. Fees, spread, slippage, **funding**, borrow, leverage, liquidation and ADL consequences **belong to the execution simulator in work package P2** and must never be folded into this label. A model that predicts this event well has demonstrated event forecasting only and must not be described as profitable.

The `price_convention` block carries five explicit `false` flags (`fees_included`, `spread_included`, `slippage_included`, `funding_included`, `leverage_included`). The evaluation plan states that a gross event metric and a net strategy return are different objects and must never be reported in the same column.

### 6.3 Event, label and the definition hash

- **Event:** `perp_forward_mark_return_up_25bps_1d_gross` — 1-day horizon, 25 bps, gross mark-to-mark return.
- **Outcome rule:** `outcome = (10000 * ((exit_mark / entry_mark) - 1)) >= 25.0`, evaluated in exact decimal arithmetic, half-open interval `[25.0, +inf)`, no epsilon.
- **Definition hash:** recomputed for this revision because the definition object changed (spot → perp, last → mark, long-only → two-sided declaration). Recorded value:

  ```text
  8796b7f90a0f62f972f0f40e80e07b100f559f47342aece0bf5582fc27392ac9
  ```

  The superseded spot-era hash `a6f06667…` is **void**. Any cohort or result carrying it is stale. This value is **not yet frozen**; R1 must recompute it from the frozen object and must match.

### 6.4 Feature allowlist — 12 perpetual observables

`mark_price`, `index_price`, `mark_index_basis_bps`, `last_funding_rate`, `funding_interval_hours`, `open_interest_level`, `open_interest_log_change_1d`, `quote_volume`, `trade_count`, `taker_buy_ratio`, `realized_vol_24bar`, `liquidation_intensity_1d` (**conditional**).

Forbidden features now include, in addition to revision 1's list: any **spot-only series used as a substitute for a perpetual observable**; any feature mixing last and mark price without declaring which; any open-interest or funding series whose **unit or sign convention is not recorded**; and **any leverage, margin-mode or position-sizing variable**.

### 6.5 Universe, holdout and seeds

- **Universe:** `crypto_perp_secondary_market_v1` — perpetuals only, USDT numeraire for V1, liquidity screen on trailing 30-bar median quote volume with an explicit warning that coin-margined volume needs a **point-in-time** USDT conversion or exclusion.
- **Instrument holdout adapted to perps:** a **venue holdout** level (one or two of the four venues withheld entirely) in addition to the hashed-identifier instrument holdout, with an explicit warning that hashing **scatters near-identical cross-venue contracts across groups** and overstates cross-sectional generalization — an underlying-clustered partition is the candidate refinement, decidable only at R1.
- **Regime holdout adapted to perps:** `regime_high_realized_volatility`, **`regime_funding_extreme`** (new, perp-only), **`regime_basis_blowout`** (new, perp-only), `regime_thin_liquidity`. Each is defined point-in-time. The protocol discloses that funding-extreme and basis-blowout are near-deterministic functions of allowlisted features, so a model may already encode the regime; the holdout tests conditional performance, not regime novelty.
- **Seeds:** unchanged and still ≥3 — **1729, 2718, 3141**. Data seed 20260919. `draft_pending_R1: true` is retained.

---

## 7. Storage and compute feasibility on an Apple Silicon Mac laptop

Review machine, measured during this review: **Apple M5 Max, 18 logical cores, 128 GB unified memory, macOS 26.5.1 arm64**. Figures below are engineering estimates from measured file sizes and stated field counts, not benchmark results.

### 7.1 Observed perp archive sizes (measured during this revision)

```text
data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2023-01.zip        863 B    (93 data rows)
data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2023-01.zip                  2,133 B   (31 rows)
data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip        22,590 B   (744 rows)
data/futures/um/monthly/indexPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip       27,296 B   (744 rows)
data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip     16,774 B   (744 rows)
data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2023-01-02.zip             14,459 B   (288 rows)
```

Note how much cheaper the **perp** datasets are than the **spot** 1-minute archive measured in revision 1 (2.3 MB/month). Perp funding is a few hundred bytes per month; OI/positioning is ~14 KB per symbol-day; hourly price references are ~20 KB per symbol-month.

### 7.2 Estimates for the proposed cohort

40 USDT-margined linear perpetual instruments × 3 years:

| Layer | Estimate | Basis |
|---|---|---|
| Daily mark/index/last klines | < 15 MB compressed, ~35 MB Parquet | ~3×1,095 symbol-days, small per-file |
| Funding (all venues) | < 10 MB compressed | 93 data rows/symbol-month at Binance; hourly on Hyperliquid ≈ 26 k rows/symbol-year |
| Open interest / metrics (5-minute) | ~2.5 GB compressed for Binance alone | 288 rows × 1,095 days × 40 symbols ≈ 12.6 M rows |
| Derived feature panel, daily | < 1 GB | 12 features × ~44 k rows × 8 bytes ≈ 4 MB before overhead |
| Hourly price references (opt.) | ~3 GB compressed | ~20 KB × 40 × 36 months |
| Hyperliquid S3 `asset_ctxs` | unverified; **egress-metered** | Monthly CSVs; cost depends on size, which was not measured |

**Conclusion.** The recommended daily stage is **trivially comfortable** (well under 1 GB including derived features). The 5-minute open-interest layer is the largest realistic commitment at a few GB compressed, still comfortable. No layer here approaches the 100 GB+ that revision 1's spot `aggTrades` estimate implied.

### 7.3 Compute

- **Feature construction** over ~44 k daily rows or ~12.6 M 5-minute OI rows is seconds to minutes of vectorized CPU work on 18 cores.
- **Deterministic and supervised baselines** on a ~44 k × 12 daily panel are seconds to low minutes on CPU. The entire B3 baseline program fits in a laptop session and needs no GPU.
- **Model training.** The existing NanoJev decision head is small and MPS FP32 works on this hardware; a bounded daily-state cohort is feasible locally.
- **The real constraint is not compute — it is label maturity, funding-time alignment, and per-venue depth reconciliation.** Aligning decision times, horizon ends, label availability, embargo, funding settlement grids and the differing venue frontiers is CPU-cheap but attention-expensive.
- **Hyperliquid is the binding constraint on any intraday extension**, not the hardware: 5,000 candles ≈ 3.5 days at 1-minute resolution.

### 7.4 Operational recommendations

1. Store raw retrievals **immutably** with recorded SHA-256 and retrieval timestamps. Binance publishes `.CHECKSUM` files alongside every archive; verify them and retain the results. This is the only point-in-time evidence obtainable for free.
2. **Normalize per venue at the ingestion boundary**: Binance futures to milliseconds→nanoseconds, and **treat `metrics.create_time` (a UTC datetime string) as a distinct parser path** from the integer-epoch klines/funding files. Bybit milliseconds with its own interval vocabulary (`D`/`W`/`M`). Aster milliseconds. Hyperliquid milliseconds.
3. Convert to Parquet partitioned by `venue/contract_type/symbol/year`; keep raw files as the audit source.
4. Assert `event_ns <= available_ns <= decision_ns` and `fit_cutoff_ns <= available_ns`, and assert **funding-rate units** (decimal vs percent), **open-interest units**, and **sign conventions** before writing any record. These assertions catch the silent-factor errors named in §5.
5. Never store a derived series without its `source_id`, `version` and `fit_cutoff_ns`.
6. Do not retain broker or venue credentials anywhere in this repository. This plan requires none, and none has been created.

---

## 8. How this plan stays compatible with the existing PIT validator

[`scripts/financial_pit_v1.py`](../scripts/financial_pit_v1.py) enforces a strict protocol shape and reads the protocol file, rejecting anything whose top-level key set is not exactly:

```python
if set(protocol) != {"folds", "embargo_ns", "asof_ns"}:
    raise ValueError("protocol requires folds, embargo_ns and asof_ns")
```

**The perpetual revision does not touch this.** `pit_validator_core` still contains **exactly** the three keys `{folds, embargo_ns, asof_ns}`. Every perpetual-specific declaration lives in sibling objects (`perp_semantics`, `event_definition`, `feature_allowlist`, `universe_construction`, holdout plans, seed plan). **No file under `scripts/` is modified.** Projecting the core to a validator input is a documented one-line operation performed by the caller.

### 8.1 The structural constraint on label lag, restated

The validator applies two exclusion rules to non-test rows, bounding the declared label availability lag from **both** sides:

1. **Maturity rule.** A non-test row is excluded when `label.available_ns > that phase's own end_ns`. Since `available_ns = decision_ns + horizon_ns + lag_ns`, a phase retains only decisions made at least `horizon_ns + lag_ns` before its own end. Each dev and calibration window must therefore be **longer than the lag plus the horizon**.
2. **Purge rule.** A row is also excluded when `label.end_ns + embargo_ns >= the next phase's start`, requiring the spacing between consecutive phase starts to exceed `lag_ns + horizon_ns + embargo_ns`.

The practical consequence, found empirically: a 30-day lag with one-month dev and calibration windows produces **entirely empty** dev and calibration phases. The protocol therefore declares a **7-day** lag, 32-day dev and calibration windows, and 33-day spacing between phase starts. This is unchanged by the perpetual revision and remains valid for 24/7 perpetual markets.

### 8.2 Measured receipt obtained during this revision

A **40-instrument synthetic perpetual-shaped cohort** (four synthetic venue tags, daily bars, cohort span `[2023-01-01, 2026-11-26)`, 6 perpetual features, 1-day horizon, 7-day lag, 1-hour embargo), conforming to the `nanojev-financial-pit-v1` record contract, was run through the **unmodified** `scripts/financial_pit_v1.py` with the projected core:

```text
$ .venv/bin/python -c "import json,pathlib; json.dump(json.loads(pathlib.Path('research/financial_experiment_protocol_v1.json').read_text())['pit_validator_core'], open('/tmp/perp_research/projected_core.json','w'), indent=2)"
$ .venv/bin/python scripts/financial_pit_v1.py \
    --input /tmp/perp_research/perp_synthetic_rows.jsonl \
    --protocol /tmp/perp_research/projected_core.json \
    --output /tmp/perp_research/perp_fold_audit.json
{"output": "/tmp/perp_research/perp_fold_audit.json", "folds": 3}

VALIDATOR EXIT = 0
projected core key set == {"folds","embargo_ns","asof_ns"}   (asserted True)

fold 1  train 31000  dev  960  calibration  960  test 7240   all_phases_nonempty True
fold 2  train 38240  dev  960  calibration  960  test 7240   all_phases_nonempty True
fold 3  train 45480  dev  960  calibration  960  test 7200   all_phases_nonempty True

exclusion reasons (identical shape in every fold):
  label_unavailable_at_fit_cutoff  880
  purged_overlap_or_embargo         80
  outside_windows                  15840 / 8600 / 1400
```

**Scope of that receipt.** It proves that the split geometry and the projected core object are structurally accepted by the unmodified validator, that all four phases are nonempty in all three folds, and that a **perpetual-shaped feature schema passes the record contract**. It proves **nothing** about real perpetual-market data, point-in-time authenticity, source licensing, funding-cost handling, feature lineage, or predictive performance. It was produced from constructed synthetic rows with a deliberately arbitrary outcome rule and must never be reported as financial evidence.

**Incident recorded for transparency.** During this revision the protocol file's `folds[1].dev[0]` was briefly corrupted to a 16-digit value (`1758672000000000` instead of `1758672000000000000`, i.e. 1970-01-21 rather than 2025-09-24), which makes the unmodified validator fail hard with `ValueError: fold windows must be nonempty, chronological and disjoint` and exit 1. The reviewed value was restored and the exit-0 receipt above was re-obtained afterwards. This is recorded because it is exactly the class of silent corruption that a hand-edited numeric window invites, and because the validator caught it.

The compatibility claim is therefore **structurally verified on synthetic perpetual-shaped data**, but **not yet evidenced on real data**. The R1 freeze must carry a receipt produced against the real frozen cohort.

---

## 9. Open questions for R1

| # | Question | Why it matters | Default if unanswered |
|---|---|---|---|
| **Q1** | Accept the four-venue **perpetual-only** scope with Binance's Bulk archive as the primary price source? | Sets the entire downstream experiment | Recommend yes |
| **Q2** | **Does this project ever intend to submit live orders?** The Binance archive's CC BY-NC-SA 4.0 terms §4.2 prohibit "live proprietary trading execution" and §3.4 excludes commercial licensing. | Determines whether the primary data source can ever support the program's end state, and whether features/labels must be re-derived later | Assume offline-only research; treat execution as blocked pending a different data right |
| **Q3** | Is "no point-in-time vintages, hash-pinned going forward only" acceptable, given the archive is **documented as having been rewritten in place**? | Determines whether the first experiment is scientifically meaningful or engineering validation | Mark every result as pipeline validation, not financial evidence |
| **Q4** | Who independently confirms the licences and terms of record, and **how are Bybit/Aster/Hyperliquid terms to be obtained given all three were unreachable from the review environment?** | Constraint 0(c) is not satisfiable by an agent reading web pages | Ingestion of the three unverified venues stays blocked |
| **Q5** | Is liquidation/ADL event history required, given **no venue publishes it as a historical series**? | Determines whether `liquidation_intensity_1d` survives | Drop the feature and record the gap |
| **Q6** | Is Aster's **missing open-interest endpoint** disqualifying for the first cohort? | Aster cannot contribute OI features | Include Aster for price/funding only, or exclude it from V1 |
| **Q7** | How is Hyperliquid's **5,000-candle API cap** and roughly-monthly, possibly-gapped S3 archive reconciled with the folds? | Hyperliquid may be daily-only | Restrict Hyperliquid to daily bars from the S3 archive |
| **Q8** | Is a single-numeraire **USDT-margined linear** cohort accepted for V1, deferring coin-margined inverse and USDC-margined on-chain perps? | The three contract families have different numeraire bases | Restrict V1 to USDT-margined linear |
| **Q9** | Is a **venue holdout** required in addition to an instrument holdout? | Cross-venue generalization evidence | Instrument holdout only; venue holdout as a pre-registered extension |
| **Q10** | Are the 25 bps / 1-day threshold and the **mark-price** labelling convention accepted? | Freezes the event definition before any result | Apply the protocol's mark-price convention |
| **Q11** | Are the three declared seeds 1729, 2718, 3141 final? | Pre-registration discipline | Treat as declared and unfrozen |
| **Q12** | Is 1-minute ingestion part of the first frozen protocol or a separately pre-registered extension, given per-venue depth differences? | Changes run time and feasibility per venue | Separately pre-registered extension |
| **Q13** | What is the policy for reconciling different open-interest grids (Binance 5-minute daily files vs Bybit's selectable grid vs Hyperliquid hourly)? | Cross-venue OI features need a common grid | Downsample to a declared common grid and record the resampling rule |

---

## 10. What this document deliberately does not do

- No dataset was purchased, subscribed to, or assembled. The only retrievals were reading public documentation and terms pages, a small number of `HEAD` probes, and **six archive files each under 30 KB** used solely to read file headers, row counts and timestamp encodings. All were written to `/tmp`, none into this repository.
- No account was created, no API key generated, no credential exists, and **no wallet deposit was made** (which is why Aster's authenticated force-orders route is out of reach).
- No broker, exchange trading endpoint, or order-submission path was contacted. No authenticated endpoint was called.
- No financial model was trained, and no result of any kind is reported.
- No profitability, edge, or performance claim is made anywhere in this document.
- **No spot instrument appears anywhere in this revision.** The track is perpetual-contract only.
- No file other than [`docs/FINANCIAL_DATA_PLAN_V1.md`](FINANCIAL_DATA_PLAN_V1.md), [`research/financial_source_manifest_v1.json`](../research/financial_source_manifest_v1.json), and [`research/financial_experiment_protocol_v1.json`](../research/financial_experiment_protocol_v1.json) was created or modified.
- No `git commit` or `git push` was run.
