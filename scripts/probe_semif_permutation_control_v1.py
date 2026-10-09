"""Control for the letter-position artifact: same weights, same readout, options PERMUTED.

If the model reads content, the chosen DESCRIPTION is stable across permutations.
If it is a position artifact, the chosen LETTER is stable (always A) and the description changes.
"""
import json, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path('/Users/markus/Documents/NanoJev/scripts')))
import importlib.util
spec = importlib.util.spec_from_file_location('probe', '/Users/markus/Documents/NanoJev/scripts/probe_semif_readout_v1.py')
# Re-implement inline to control option order; reuse loading logic from the probe module's top half.
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

ROOT = pathlib.Path('/Users/markus/Documents/NanoJev')
CKPT = ROOT/'checkpoints/local_atomic_seed17/variants/local_atomic_seed17'
SYS = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
       "Respond with only its uppercase letter, with no explanation or reasoning.")
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(str(CKPT/'tokenizer'), local_files_only=True)
cfg = AutoConfig.from_pretrained(str(CKPT/'backbone_config'), local_files_only=True)
model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
sd = load_file(str(CKPT/'best.safetensors'))
model.load_state_dict({'model.'+k[len('backbone.'):]: v for k,v in sd.items() if k.startswith('backbone.')}, strict=False)

def slot(letter):
    ids = tok.encode(letter, add_special_tokens=False)
    assert len(ids)==1 and tok.decode(ids)==letter
    return ids[0]

def ask(state, instr, descs):
    letters = [chr(65+i) for i in range(len(descs))]
    payload = {'evidence': state, 'criterion': instr,
               'options': [{'letter': L, 'description': d} for L, d in zip(letters, descs)]}
    msgs = [{'role':'system','content':SYS},{'role':'user','content':json.dumps(payload, ensure_ascii=False)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    lg = model(**tok(text, return_tensors='pt'), use_cache=False).logits[0,-1,:]
    p = torch.softmax(lg[[slot(L) for L in letters]], dim=-1)
    i = int(torch.argmax(p))
    return letters[i], descs[i], float(p[i])

def base_options(q):
    if q['type']=='choice': return [f'{k}: {v}' for k,v in q['criteria'].items()]
    if q['type']=='score': return list(q['criteria'])
    return ['The proposition is true.', 'The proposition is false.']

survey = json.loads((ROOT/'research/skill_attendance_survey.json').read_text()) if False else json.loads((ROOT/'research/skill_abstention_survey_v1.json').read_text())
print(f"{'question':16s} {'letter':>7s} {'chosen description (orig | reversed)':s}")
letter_stable = desc_stable = 0
n = 0
for st in survey['request']['states']:
    for qid, q in st['questions'].items():
        opts = base_options(q)
        L1, D1, C1 = ask(st['state'], q['instructions'], opts)
        L2, D2, C2 = ask(st['state'], q['instructions'], list(reversed(opts)))
        # map reversed back to the original description identity
        same_desc = (D1 == D2)
        same_letter = (L1 == L2)
        letter_stable += same_letter; desc_stable += same_desc; n += 1
        print(f"  {qid:16s} {L1}/{L2:>5s}  {'SAME' if same_desc else 'DIFF'}  {D1[:34]} | {D2[:30]}")
print(f"\nchosen LETTER identical across permutation: {letter_stable}/{n}")
print(f"chosen DESCRIPTION identical across permutation: {desc_stable}/{n}")
print("=> position artifact" if desc_stable < n else "=> content-driven")
