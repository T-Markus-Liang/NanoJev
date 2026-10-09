"""Attribute the confidence collapse: content domain vs surface form (length, wording).

Read-only inference against the already-running local service. No training, no checkpoint writes.
"""
import json, pathlib, subprocess, sys

ROOT = pathlib.Path('/Users/markus/Documents/NanoJev')
HELPER = '/Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py'

maze = [json.loads(l) for l in (ROOT/'dataset/games_v4/data/local_maze_v1/test.jsonl').read_text().splitlines() if l.strip()][:4]
survey = json.loads((ROOT/'research/skill_abstention_survey_v1.json').read_text())['request']['states']

# An irrelevant but plausible engineering paragraph, used ONLY to lengthen an in-domain state.
PAD = (" Repository context (not relevant to this question): the working tree carries an in-flight "
       "patch to the execution simulator, a stashed variant of the paper-trading driver from a "
       "concurrent writer, a frozen protocol JSON, three fetch manifests with per-file hashes, and "
       "a review log whose most recent entry records a measurement defect. The suite reports 411 "
       "passing tests and two skipped. Nothing in this paragraph bears on the map below.")

def eng_style(text):
    """Rewrite maze wording into engineering prose without changing meaning or length class."""
    return ("Holding the map fixed, decide the following proposition about the agent's immediate "
            "environment. " + text.replace("The destination of the one-cell", "The target cell of the single-step"))

states = []
# --- in-domain content, four surface forms ---
for i, row in enumerate(maze):
    qs = row['questions']
    orig = {qid: {'type': q['type'], 'instructions': q['instructions'], 'criteria': q.get('criteria', {})}
            for qid, q in qs.items()}
    reword = {qid: {'type': q['type'], 'instructions': eng_style(q['instructions']),
                    'criteria': {k: eng_style(v) for k, v in (q.get('criteria') or {}).items()}}
              for qid, q in qs.items()}
    states.append({'id': f'maze{i}_v0_plain', 'state': row['state'], 'questions': orig})
    states.append({'id': f'maze{i}_v1_padded', 'state': row['state'] + PAD, 'questions': orig})
    states.append({'id': f'maze{i}_v2_reworded', 'state': row['state'], 'questions': reword})
    states.append({'id': f'maze{i}_v3_both', 'state': row['state'] + PAD, 'questions': reword})

# --- out-of-domain content, shortened/plain surface form ---
short = {
 'phase': ("B0 is not done. Gateway and A4 are delivered. Which phase next?",
           {'next_phase': {'type':'choice','instructions':'Pick the next phase.',
             'criteria':{'development':'write code','testing':'run tests','optimization':'tune speed','deployment':'ship it'}}}),
 'routing': ("A history has 400 segments; the gate scores at most 32.",
           {'route': {'type':'choice','instructions':'Who handles it?',
             'criteria':{'main_model':'send all context','gate_bypass':'let the gate bypass','batch_merge':'batch and merge'}}}),
}
for name,(st,qs) in short.items():
    states.append({'id': f'eng_{name}_v0_plain', 'state': st, 'questions': qs})

req = {'states': states}
nq = sum(len(s['questions']) for s in states)
npaths = sum(1 if q['type']=='boolean' else len(q['criteria']) for s in states for q in s['questions'].values())
print(f'states={len(states)} questions={nq} paths={npaths}')
assert len(states)<=32 and nq<=96 and npaths<=256, 'exceeds service limits'
pathlib.Path('/tmp/attribution_request.json').write_text(json.dumps(req, indent=2))

out = subprocess.run([sys.executable, HELPER, 'decide', '--input', '/tmp/attribution_request.json',
                      '--source','codex','--task-tag','attribution_probe'],
                     capture_output=True, text=True)
if out.returncode != 0:
    print('FAILED', out.returncode, out.stderr[-500:]); raise SystemExit(1)
d = json.loads(out.stdout)
res = {}
for st in d['states']:
    for qid, a in st['answers'].items():
        res.setdefault(st['id'], []).append((a['confidence'], a['type']))
print('\n--- median confidence by variant ---')
import statistics
rowsout=[]
for sid in [s['id'] for s in states]:
    cs=[c for c,_ in res[sid]]
    rowsout.append((sid, statistics.median(cs), min(cs), max(cs), len(cs)))
for sid, med, lo, hi, n in rowsout:
    print(f'  {sid:26s} median={med:.3f} min={lo:.3f} max={hi:.3f} n={n}')
json.dump({'states':[s['id'] for s in states],
           'median':{sid:med for sid,med,_,_,_ in rowsout},
           'raw':{k:[c for c,_ in v] for k,v in res.items()},
           'event_id': d['usage']['event_id']},
          open('/tmp/attribution_result.json','w'), indent=2)
print('\nevent:', d['usage']['event_id'])
