"""Read the SAME checkpoint's decisions via SemIf/JEV-CPU's letter-choice logit readout.

Motivation: our trained-head readout abstains 13/13 on engineering questions while the base model
is Qwen3-0.6B. SemIf (TheoLeeCJ/SemIf, 1770 stars) reads P(option letter | evidence, criterion,
options) from ONE forward pass with no task-fitted head, and JEV-CPU demonstrates it across eight
domains on this exact base model. This probe holds the weights fixed and changes only the readout.
"""
import json, pathlib, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / 'checkpoints/local_atomic_seed17/variants/local_atomic_seed17'
SYS = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
       "Respond with only its uppercase letter, with no explanation or reasoning.")

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(str(CKPT / 'tokenizer'), local_files_only=True)
cfg = AutoConfig.from_pretrained(str(CKPT / 'backbone_config'), local_files_only=True)
model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
sd = load_file(str(CKPT / 'best.safetensors'))
# The checkpoint stores the base model under a 'backbone.' prefix; Qwen3ForCausalLM expects
# the same tensors under 'model.'. Map one to the other so the weights are actually loaded.
body = {'model.' + k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
missing, unexpected = model.load_state_dict(body, strict=False)
print(f'backbone tensors loaded: {len(body)} | missing={len(missing)} unexpected={len(unexpected)}')
if missing: print('  missing sample:', list(missing)[:3])

def options_for(q):
    """Assign single letters to options; the criteria keys in our survey are words, not letters."""
    t = q['type']
    if t == 'choice':
        pairs = list(q['criteria'].items())
        return [{'letter': chr(65 + i), 'description': f'{k}: {d}'} for i, (k, d) in enumerate(pairs)]
    if t == 'score':
        return [{'letter': chr(65 + i), 'description': d} for i, d in enumerate(q['criteria'])]
    # boolean: our survey carries no criteria, so declare the two poles explicitly
    return [{'letter': 'A', 'description': 'The proposition is true.'},
            {'letter': 'B', 'description': 'The proposition is false.'}]

def slot_id(letter):
    ids = tok.encode(letter, add_special_tokens=False)
    assert len(ids) == 1 and tok.decode(ids) == letter, f'{letter} is not a single clean token: {ids}'
    return ids[0]

def decide(state, q):
    opts = options_for(q)
    payload = {'evidence': state, 'criterion': q['instructions'],
               'options': [{'letter': o['letter'], 'description': o['description']} for o in opts]}
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    inputs = tok(text, return_tensors='pt')
    logits = model(**inputs, use_cache=False).logits[0, -1, :]
    slots = [slot_id(o['letter']) for o in opts]
    probs = torch.softmax(logits[slots], dim=-1)
    best = int(torch.argmax(probs))
    return opts[best]['letter'], opts[best]['description'], float(probs[best]), [round(float(p), 4) for p in probs]

survey = json.loads((ROOT / 'research/skill_abstention_survey_v1.json').read_text())
rows = []
for st in survey['request']['states']:
    for qid, q in st['questions'].items():
        letter, desc, conf, probs = decide(st['state'], q)
        rows.append((st['id'], qid, letter, desc, conf))
        print(f'  {st["id"]:9s} {qid:16s} conf={conf:.3f}  {letter}: {desc[:40]}')

confs = [r[4] for r in rows]
import statistics
print(f'\n=== SemIf letter-choice readout on the same checkpoint ===')
print(f'confidence: min={min(confs):.3f} median={statistics.median(confs):.3f} max={max(confs):.3f}')
for th in (0.9, 0.7, 0.5):
    print(f'  answered at {th}: {sum(1 for c in confs if c >= th)}/{len(confs)}')
json.dump({'readout': 'semif-letter-choice-logits', 'checkpoint': str(CKPT),
           'rows': [{'state': a, 'qid': b, 'letter': c, 'answer': d, 'confidence': e}
                    for a, b, c, d, e in rows]},
          open(ROOT / 'results/semif_readout_probe_v1.json', 'w'), indent=2)
