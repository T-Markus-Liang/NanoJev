import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

// Evaluate any local TypeSafe-compatible /v1/systemone endpoint on the frozen
// offline bundle. This writes a receipt shaped like eval_official_jev_bundle_v1.mjs
// so score_official_jev_bundle_v1.py can normalize it. Evaluation-only: endpoint
// outputs are never written into training data.

const ROOT = new URL('../', import.meta.url).pathname;
const BUNDLE = `${ROOT}data/jevbench_offline_bundle_v1`;
const args = process.argv.slice(2);
const arg = (name, fallback = null) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const baseUrl = (arg('base-url', 'http://127.0.0.1:8010')).replace(/\/$/, '');
const model = arg('model', 'local');
const receipt = arg('receipt', `${ROOT}results/systemone_${model.replace(/[^A-Za-z0-9_.-]/g, '_')}.jsonl`);
const summaryPath = arg('summary', receipt.replace(/\.jsonl$/, '.summary.json'));
const limit = args.includes('--limit') ? Number(arg('limit')) : null;
const trackFilter = args.includes('--tracks') ? new Set(arg('tracks').split(',')) : null;
const concurrency = Number(arg('concurrency', '1'));
const timeoutMs = Number(arg('timeout-ms', '120000'));
const apiKeyEnv = arg('api-key-env', null);
const allowRemote = args.includes('--allow-remote');
const dryRun = args.includes('--dry-run');

if (!allowRemote && !/^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/.test(baseUrl)) {
  throw new Error(`refusing non-local endpoint ${baseUrl}; pass --allow-remote if intentional`);
}

const readJsonl = async (path) =>
  (await readFile(path, 'utf8')).split('\n').filter(Boolean).map(JSON.parse);

async function loadRows() {
  const rows = [];
  for (const file of ['original', 'easy', 'hard']) {
    for (const r of await readJsonl(`${BUNDLE}/official_jevbench_v1.2.4_public/${file}.jsonl`)) {
      const q = r.question;
      rows.push({
        track: 'official_jevbench_public', id: r.id, family: r.family,
        state: r.state, type: q.type, instructions: q.instructions,
        criteria: q.criteria, gold: r.expected,
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
  return Number(Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0]?.[0]);
}

async function runOne(row) {
  const question = { type: row.type, instructions: row.instructions };
  if (row.criteria !== undefined) question.criteria = row.criteria;
  const headers = { 'Content-Type': 'application/json' };
  if (apiKeyEnv) headers.Authorization = `Bearer ${process.env[apiKeyEnv] ?? ''}`;
  const start = performance.now();
  const res = await fetch(`${baseUrl}/v1/systemone`, {
    method: 'POST', headers,
    body: JSON.stringify({ state: row.state, model, questions: { q: question } }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const elapsedMs = performance.now() - start;
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    const err = new Error(`systemone ${res.status}: ${body.slice(0, 300)}`);
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
    model: result.model ?? model,
    ts: new Date().toISOString(),
  };
}

const all = await loadRows();
const selected = all.filter(r => !trackFilter || trackFilter.has(r.track));
const done = new Set();
if (existsSync(receipt)) {
  for (const l of (await readFile(receipt, 'utf8')).split('\n').filter(Boolean)) {
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
console.log(JSON.stringify({ mode: dryRun ? 'dry-run' : 'live-local', baseUrl, model,
  rows_total: all.length, selected: selected.length, already_done: done.size,
  pending: capped.length, concurrency, receipt, byTrack }, null, 2));
if (dryRun || capped.length === 0) process.exit(0);

await mkdir(new URL('../results/', import.meta.url).pathname, { recursive: true });
await mkdir(new URL('../research/', import.meta.url).pathname, { recursive: true });
const queue = [...capped];
let completed = 0;
await Promise.all(Array.from({ length: concurrency }, async () => {
  while (queue.length) {
    const row = queue.shift();
    try {
      const rec = await runOne(row);
      await appendFile(receipt, JSON.stringify(rec) + '\n');
    } catch (e) {
      await appendFile(receipt, JSON.stringify({
        track: row.track, id: row.id,
        error: { name: e.name, message: e.message, status: e.statusCode ?? null },
      }) + '\n');
    }
    if (++completed % 25 === 0)
      console.log(JSON.stringify({ completed, of: capped.length }));
  }
}));

const latest = new Map();
for (const l of (await readFile(receipt, 'utf8')).split('\n').filter(Boolean)) {
  const r = JSON.parse(l);
  latest.set(`${r.track}/${r.id}`, r);
}
const recs = [...latest.values()];
const lat = recs.filter(r => r.elapsed_ms != null).map(r => r.elapsed_ms).sort((a, b) => a - b);
const pct = p => lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null;
const groups = {};
for (const r of recs) {
  for (const key of [r.track, `family:${r.family}`]) {
    const g = (groups[key] ??= { n: 0, scored: 0, correct: 0, errors: 0 });
    g.n++;
    if (r.error) { g.errors++; continue; }
    if (r.correct !== null) { g.scored++; g.correct += r.correct ? 1 : 0; }
  }
}
for (const g of Object.values(groups))
  g.accuracy = g.scored ? +(g.correct / g.scored).toFixed(4) : null;
const summary = {
  ts: new Date().toISOString(), channel: 'local-systemone', base_url: baseUrl,
  requested_model: model, reported_models: [...new Set(recs.map(r => r.model).filter(Boolean))],
  receipt, calls: recs.length, errors: recs.filter(r => r.error).length,
  latency_ms: { p50: pct(0.5), p95: pct(0.95), n: lat.length },
  tokens_in: recs.reduce((n, r) => n + (r.input_tokens ?? 0), 0),
  tokens_out: recs.reduce((n, r) => n + (r.output_tokens ?? 0), 0),
  groups,
};
await writeFile(summaryPath, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ done: completed, summary: summaryPath }));
