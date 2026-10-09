import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { evaluateTeacher } from './teachers.mjs';

// Generic paired-evaluation runner for official Jev via the direct TypeSafe API
// (api.typesafe.ai, model jev-latest). Input: JSONL rows
// {id, state, questions:{qid:{type,instructions,criteria}}}
// Output: responses JSONL + summary JSON. Budget-capped; bounded 429/5xx
// backoff; records raw answers, token usage and per-request latency.
// Raw responses stay in results/ (aggregate + per-row answers; no secrets/headers).

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? true] : [a, true];
}));
const input = args.input;
const output = args.output ?? 'results/jev_batch_v1';
const budget = parseFloat(args.budget ?? '2.00');
const limit = args.limit ? parseInt(args.limit) : Infinity;
if (!input) { console.error('usage: node jev_batch_eval.mjs --input=rows.jsonl [--output=dir] [--budget=2.00] [--limit=N]'); process.exit(2); }

const PRICE_PER_TOKEN = 0.000000042;
const lines = (await readFile(input, 'utf8')).split('\n').filter(l => l.trim());
const rows = lines.map(l => JSON.parse(l)).slice(0, limit);
await mkdir(output, { recursive: true });
const outPath = `${output}/responses.jsonl`;

let usedUsd = 0, done = 0, errors = 0;
const t0 = Date.now();
const out = [];
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function callJev(row) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const t = Date.now();
    try {
      const r = await evaluateTeacher({
        teacher: 'jev', model: 'jev-latest', state: row.state, questions: row.questions,
      });
      return { r, elapsed: Date.now() - t, attempts: attempt + 1 };
    } catch (e) {
      const status = e.statusCode ?? e.cause?.statusCode;
      if ((status === 429 || status === 500 || status === 503 || status === 504) && attempt < 5) {
        await sleep(4000 * (attempt + 1));
        continue;
      }
      throw e;
    }
  }
}

for (const row of rows) {
  try {
    const { r, elapsed, attempts } = await callJev(row);
    const rec = { id: row.id, elapsed_ms: elapsed, attempts, input_tokens: r.usage?.inputTokens ?? null,
      output_tokens: r.usage?.outputTokens ?? null, model: r.model ?? null,
      answers: r.answers ?? null, rounding: r.rounding ?? null };
    out.push(JSON.stringify(rec)); done++;
    usedUsd += (r.usage?.inputTokens ?? 0) * PRICE_PER_TOKEN;
  } catch (e) {
    out.push(JSON.stringify({ id: row.id, error: { name: e.name, status: e.statusCode ?? null, message: String(e.message).slice(0, 300) } }));
    errors++;
  }
  if (usedUsd >= budget) { out.push(JSON.stringify({ aborted: 'budget', used_usd: usedUsd })); break; }
}
await writeFile(outPath, out.join('\n') + '\n');

const inputHash = createHash('sha256').update(lines.join('\n')).digest('hex');
const summary = { schema_version: 'nanojev-jev-batch-eval-v1', timestamp: new Date().toISOString(),
  channel: 'typesafe-direct', model: 'jev-latest', input, input_sha256: inputHash, rows: rows.length,
  completed: done, errors, elapsed_seconds: (Date.now() - t0) / 1000,
  estimated_cost_usd_total: +usedUsd.toFixed(6), budget_cap_usd: budget,
  caveat: 'Paired-run responses for scoring against frozen gold. Cost estimated from token counts; not a capability claim by itself.' };
await writeFile(`${output}/summary.json`, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary));
