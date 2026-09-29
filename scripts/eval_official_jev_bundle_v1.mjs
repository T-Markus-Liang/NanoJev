import { readFile, appendFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';

// Offline-bundle evaluation of official Jev via the direct TypeSafe API
// (https://api.typesafe.ai/v1/systemone, model jev-latest). The Vercel AI
// Gateway channel was removed; direct is the only path. Default is a dry run;
// --live performs paid provider calls. Receipts go to research/official_*
// which is gitignored; raw provider outputs stay local. Every track remains
// evaluation-only: outputs never enter training data.

const ROOT = new URL('../', import.meta.url).pathname;
const BUNDLE = `${ROOT}data/jevbench_offline_bundle_v1`;
const args = process.argv.slice(2);
const live = args.includes('--live');
const RECEIPT = `${ROOT}research/official_jev_direct_run_v1.jsonl`;
const SUMMARY = `${ROOT}research/official_jev_direct_summary_v1.json`;
const DIRECT_URL = 'https://api.typesafe.ai/v1/systemone';
const PRICE_PER_TOKEN = 0.000000042; // USD, Jev list price 2026-09-17
const BUDGET_USD = Number(process.env.JEV_BUDGET_USD ?? 2.0);
const CONCURRENCY = Number(process.env.JEV_CONCURRENCY ?? 2);
const RATE_LIMIT_BACKOFF_MS = [5_000, 15_000, 30_000];
const TIMEOUT_MS = 45_000;

const limit = args.includes('--limit') ? Number(args[args.indexOf('--limit') + 1]) : null;
const trackFilter = args.includes('--tracks')
  ? new Set(args[args.indexOf('--tracks') + 1].split(','))
  : null;

const readJsonl = async (path) =>
  (await readFile(path, 'utf8')).split('\n').filter(Boolean).map(JSON.parse);

async function loadRows() {
  const rows = [];
  for (const file of ['original', 'easy', 'hard']) {
    for (const r of await readJsonl(`${BUNDLE}/official_jevbench_v1.2.4_public/${file}.jsonl`)) {
      const q = r.question;
      const type = q.type;
      rows.push({
        track: 'official_jevbench_public', id: r.id, family: r.family,
        state: r.state, type, instructions: q.instructions, criteria: q.criteria,
        gold: r.expected,
      });
    }
  }
  const semifChoice = (r, track) => ({
    track, id: r.id, family: r.family, state: r.state, type: 'choice',
    instructions: r.question,
    criteria: Object.fromEntries(r.options.map(o => [o.id, o.description ?? o.id])),
    gold: r.label === undefined || r.label === null ? null : r.options[r.label].id,
  });
  for (const r of await readJsonl(`${BUNDLE}/semif_owned/authored144.jsonl`))
    rows.push(semifChoice(r, 'semif_authored144'));
  for (const r of await readJsonl(`${BUNDLE}/semif_owned/perturbations108.jsonl`))
    rows.push(semifChoice(r, 'semif_perturbations108'));
  for (const r of await readJsonl(`${BUNDLE}/semif_owned/shape777.jsonl`))
    rows.push(semifChoice(r, 'semif_shape777_nogold'));
  for (const r of await readJsonl(`${BUNDLE}/semif_rebuilt_external/wanli256.jsonl`))
    rows.push(semifChoice(r, 'semif_wanli256'));
  const gold154 = new Map(
    (await readJsonl(`${BUNDLE}/semif_rebuilt_external/every_rows/gold154.jsonl`))
      .map(r => [r.id, r.options[r.label].id]),
  );
  for (const r of await readJsonl(`${BUNDLE}/semif_rebuilt_external/every_rows/inference204.jsonl`)) {
    const row = semifChoice(r, 'semif_every204');
    row.gold = gold154.get(r.id) ?? null;
    rows.push(row);
  }
  return rows;
}

function predict(row, answer) {
  if (row.type === 'boolean' || row.type === 'noul') {
    const p = answer.noul ?? answer.probability;
    return p >= 0.5 ? 'yes' : 'no';
  }
  if (row.type === 'choice') {
    return Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0]?.[0];
  }
  // score: probabilities keyed by level index strings
  return Number(Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0]?.[0]);
}

function directApiKey() {
  if (process.env.TYPESAFE_API_KEY) return process.env.TYPESAFE_API_KEY;
  try {
    const k = readFileSync(
      `${process.env.HOME}/.config/jev-eval/apikey`, 'utf8').trim();
    if (k.startsWith('apikey_')) return k;
  } catch { /* fall through */ }
  throw new Error('TYPESAFE_API_KEY missing and ~/.config/jev-eval/apikey unavailable');
}

async function runOne(row) {
  const question = { type: row.type, instructions: row.instructions };
  if (row.criteria !== undefined) question.criteria = row.criteria;
  const start = performance.now();
  const res = await fetch(DIRECT_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${directApiKey()}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ state: row.state, model: 'jev-latest', questions: { q: question } }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const elapsedMs = performance.now() - start;
  if (!res.ok) {
    const err = new Error(`typesafe direct ${res.status}`);
    err.statusCode = res.status;
    throw err;
  }
  const result = await res.json();
  const pred = predict(row, result.answers.q);
  return {
    track: row.track, id: row.id, family: row.family, gold: row.gold, pred,
    correct: row.gold === null ? null : pred === row.gold,
    answer: result.answers.q,
    elapsed_ms: Math.round(elapsedMs),
    input_tokens: result.usage?.input_tokens ?? null,
    output_tokens: result.usage?.output_tokens ?? null,
    model: result.model ?? 'jev-latest',
    ts: new Date().toISOString(),
  };
}

const all = await loadRows();
const selected = all.filter(r => !trackFilter || trackFilter.has(r.track));
const done = new Set();
if (existsSync(RECEIPT)) {
  for (const l of (await readFile(RECEIPT, 'utf8')).split('\n').filter(Boolean)) {
    const r = JSON.parse(l);
    if (!r.error) done.add(`${r.track}/${r.id}`);
  }
}
const pending = selected.filter(r => !done.has(`${r.track}/${r.id}`));
const capped = limit === null ? pending : pending.slice(0, limit);

const byTrack = {};
for (const r of selected) {
  byTrack[r.track] ??= { total: 0, labeled: 0 };
  byTrack[r.track].total++;
  if (r.gold !== null) byTrack[r.track].labeled++;
}
const estCostUsd = capped.reduce(
  (n, r) => n + (Buffer.byteLength(JSON.stringify(r.state)) + 4096) * PRICE_PER_TOKEN, 0,
);
console.log(JSON.stringify({
  mode: live ? 'live' : 'dry-run', rows_total: all.length, selected: selected.length,
  already_done: done.size, pending: capped.length, estimated_cost_usd: +estCostUsd.toFixed(4),
  budget_usd: BUDGET_USD, concurrency: CONCURRENCY, byTrack,
}, null, 2));
if (!live || capped.length === 0) process.exit(0);
directApiKey(); // fail fast if no TypeSafe key is resolvable
if (estCostUsd > BUDGET_USD) throw new Error(`estimate ${estCostUsd} exceeds budget ${BUDGET_USD}`);

await mkdir(`${ROOT}research`, { recursive: true });
let completed = 0, spent = 0;
const queue = [...capped];
const workers = Array.from({ length: CONCURRENCY }, async () => {
  while (queue.length) {
    const row = queue.shift();
    let attempt = 0;
    for (;;) {
      try {
        const rec = await runOne(row);
        spent += (rec.input_tokens ?? 0) * PRICE_PER_TOKEN;
        await appendFile(RECEIPT, JSON.stringify(rec) + '\n');
        break;
      } catch (e) {
        const status = e.statusCode ?? e.cause?.statusCode ?? null;
        if (status === 401 || status === 403) {
          console.error('auth failure, aborting run');
          process.exit(2);
        }
        if (status === 429 && attempt < RATE_LIMIT_BACKOFF_MS.length) {
          await new Promise(r => setTimeout(r, RATE_LIMIT_BACKOFF_MS[attempt++]));
          continue;
        }
        await appendFile(RECEIPT, JSON.stringify({
          track: row.track, id: row.id, error: { name: e.name, status },
        }) + '\n');
        break;
      }
    }
    if (++completed % 50 === 0) {
      console.log(JSON.stringify({ completed, of: capped.length, spent_usd: +spent.toFixed(4) }));
    }
  }
});
await Promise.all(workers);

// score from the full receipt log (includes earlier partial runs);
// later records supersede earlier ones so retried errors do not double-count
const latest = new Map();
for (const l of (await readFile(RECEIPT, 'utf8')).split('\n').filter(Boolean)) {
  const r = JSON.parse(l);
  latest.set(`${r.track}/${r.id}`, r);
}
const recs = [...latest.values()];
const lat = recs.filter(r => r.elapsed_ms != null).map(r => r.elapsed_ms).sort((a, b) => a - b);
const pct = p => (lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null);
const groups = {};
for (const r of recs) {
  for (const key of [r.track, `family:${r.family}`]) {
    const g = (groups[key] ??= { n: 0, scored: 0, correct: 0, errors: 0 });
    g.n++;
    if (r.error) { g.errors++; continue; }
    if (r.correct !== null) { g.scored++; g.correct += r.correct ? 1 : 0; }
  }
}
for (const g of Object.values(groups)) g.accuracy = g.scored ? +(g.correct / g.scored).toFixed(4) : null;
const reportedModels = [...new Set(recs.map(r => r.model).filter(Boolean))];
const summary = {
  ts: new Date().toISOString(),
  channel: 'typesafe-direct',
  model: reportedModels[0] ?? 'jev-latest',
  reported_models: reportedModels,
  receipt: RECEIPT,
  calls: recs.length, errors: recs.filter(r => r.error).length,
  latency_ms: { p50: pct(0.5), p95: pct(0.95), n: lat.length },
  tokens_in: recs.reduce((n, r) => n + (r.input_tokens ?? 0), 0),
  spent_usd: null,
  estimated_cost_usd:
    +(recs.reduce((n, r) => n + (r.input_tokens ?? 0), 0) * PRICE_PER_TOKEN).toFixed(4),
  groups,
  notes: [
    'typesafe_selected_102 track absent: source snapshots not redistributable, excluded by bundle manifest.',
    'shape777 and every action-firewall rows have no gold labels; answers recorded but not accuracy-scored.',
    'Provider outputs are private receipts; publish aggregates only.',
  ],
};
await writeFile(SUMMARY, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ done: completed, spent_usd: summary.spent_usd, summary: SUMMARY }));
