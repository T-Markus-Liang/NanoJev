"""Run the SAME 13 engineering-judgment questions through laya (421M RLCD encoder) that NanoJev
abstains on, to test whether the collapse is NanoJev-specific or class-wide."""
import json, pathlib, statistics
import laya

survey = json.loads(pathlib.Path('/Users/markus/Documents/NanoJev/research/skill_abstention_survey_v1.json').read_text())
states = survey['request']['states']

agent = laya.load("convaiinnovations/laya")

# laya uses noul where we use boolean; criteria for boolean is optional {false,true}
def conv(q):
    t = q['type']
    out = {'type': 'noul' if t == 'boolean' else t, 'instructions': q['instructions']}
    if t == 'boolean':
        if q.get('criteria'):
            out['criteria'] = q['criteria']
    else:
        out['criteria'] = q['criteria']
    return out

rows = []
total = 0
for st in states:
    qs = {qid: conv(q) for qid, q in st['questions'].items()}
    r = agent.predict(st['state'], qs)
    ans = r['answers']
    for qid, a in ans.items():
        conf = a.get('confidence')
        if conf is None:  # noul returns a probability
            p = a.get('noul')
            conf = max(p, 1 - p) if isinstance(p, (int, float)) else None
        rows.append((st['id'], qid, conf, a.get('choice', a.get('noul', a.get('score')))))
        total += 1

print(f"\n=== laya on the SAME {total} engineering questions ===")
confs = [c for _, _, c, _ in rows if c is not None]
for sid, qid, c, v in rows:
    print(f"  {sid:9s} {qid:16s} conf={c:.3f}  -> {str(v)[:34]}")
print(f"\nconfidence: min={min(confs):.3f} median={statistics.median(confs):.3f} max={max(confs):.3f}")
for th in (0.9, 0.7, 0.5):
    print(f"  answered at {th}: {sum(1 for c in confs if c >= th)}/{len(confs)}")
json.dump({'rows': rows, 'confs': confs}, open('/tmp/laya_probe_result.json', 'w'), indent=2, default=str)
