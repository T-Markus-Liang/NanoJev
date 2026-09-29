#!/usr/bin/env python3
"""Run public Jev-class native APIs over the frozen offline bundle.

The output receipt intentionally matches eval_systemone_bundle_v1.mjs so it can
be normalized by score_official_jev_bundle_v1.py. The upstream engine is used
unmodified; only request/schema translation is recorded in adapter fields.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "jevbench_offline_bundle_v1"
TRACKS = [
    "official_jevbench_public",
    "semif_authored144",
    "semif_perturbations108",
    "semif_shape777_nogold",
    "semif_wanli256",
    "semif_every204",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def criteria_pairs(criteria: Any) -> list[tuple[str, str]]:
    if isinstance(criteria, dict):
        return [(str(k), str(v if v is not None else k)) for k, v in criteria.items()]
    if isinstance(criteria, list):
        return [(str(i), str(v)) for i, v in enumerate(criteria)]
    return []


def abstain_result(row: dict, reason: str, input_tokens: int | None = None) -> tuple[None, dict]:
    pairs = criteria_pairs(row["criteria"])
    if row["type"] == "noul":
        probs = {"no": 0.5, "yes": 0.5}
    elif row["type"] == "score":
        probs = {str(i): 1.0 / len(pairs) for i, _ in pairs}
    else:
        probs = {idx: 1.0 / len(pairs) for idx, _ in pairs}
    return None, {"probabilities": probs,
                  "adapter": "input rejected by upstream contract; mapped to explicit abstention",
                  "rejected_reason": reason,
                  "input_tokens": input_tokens}


def load_rows() -> list[dict]:
    rows = []
    official = BUNDLE / "official_jevbench_v1.2.4_public"
    for source in ("original", "easy", "hard"):
        for row in read_jsonl(official / f"{source}.jsonl"):
            q = row["question"]
            rows.append({
                "track": "official_jevbench_public",
                "id": row["id"],
                "family": row.get("family"),
                "state": row["state"],
                "type": q["type"],
                "instructions": q["instructions"],
                "criteria": q.get("criteria"),
                "gold": row.get("expected"),
            })

    def semif(row: dict, track: str) -> dict:
        return {
            "track": track,
            "id": row["id"],
            "family": row.get("family"),
            "state": row["state"],
            "type": "choice",
            "instructions": row["question"],
            "criteria": {o["id"]: o.get("description") or o["id"] for o in row["options"]},
            "gold": None if row.get("label") is None else row["options"][row["label"]]["id"],
        }

    for row in read_jsonl(BUNDLE / "semif_owned/authored144.jsonl"):
        rows.append(semif(row, "semif_authored144"))
    for row in read_jsonl(BUNDLE / "semif_owned/perturbations108.jsonl"):
        rows.append(semif(row, "semif_perturbations108"))
    for row in read_jsonl(BUNDLE / "semif_owned/shape777.jsonl"):
        rows.append(semif(row, "semif_shape777_nogold"))
    for row in read_jsonl(BUNDLE / "semif_rebuilt_external/wanli256.jsonl"):
        rows.append(semif(row, "semif_wanli256"))

    gold154 = {
        row["id"]: row["options"][row["label"]]["id"]
        for row in read_jsonl(BUNDLE / "semif_rebuilt_external/every_rows/gold154.jsonl")
    }
    for row in read_jsonl(BUNDLE / "semif_rebuilt_external/every_rows/inference204.jsonl"):
        rec = semif(row, "semif_every204")
        rec["gold"] = gold154.get(row["id"])
        rows.append(rec)
    return rows


class VerdictAdapter:
    def __init__(self, model: str, device: str):
        sys.path.insert(0, str(ROOT / "external/openjev-verdict"))
        from core.engine_encoder import DecisionEngine
        from core.primitives import Choice, Level, Noul, Option, Score
        self.Choice, self.Level, self.Noul, self.Option, self.Score = Choice, Level, Noul, Option, Score
        self.engine = DecisionEngine(model, device=device)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            q = self.Noul(id="q", proposition=row["instructions"],
                          semantics="conditional_on_sufficient_evidence_v2")
        elif row["type"] == "score":
            q = self.Score(id="q", question=row["instructions"], levels=tuple(
                self.Level(id=idx, description=text, value=float(idx))
                for idx, text in pairs))
        else:
            q = self.Choice(id="q", question=row["instructions"], options=tuple(
                self.Option(id=idx, description=text) for idx, text in pairs))
        result = self.engine.evaluate(state_text(row["state"]), [q]).results[0]
        probs = result.probabilities
        if row["type"] == "noul":
            pred = {"true": "yes", "false": "no"}.get(result.selected_outcome,
                                                     result.selected_outcome)
        elif row["type"] == "score":
            pred = int(result.selected_level_id) if result.selected_level_id and result.selected_level_id.isdigit() else result.selected_level_id
        else:
            pred = result.selected_id
        return pred, {"probabilities": probs, "adapter": "openjev-verdict native"}


class OpenJevAdapter:
    def __init__(self, model: str, device: str):
        sys.path.insert(0, str(ROOT / "external/typed-decisions/src"))
        from typed_decisions.open_jev import OpenJev
        self.model = OpenJev.from_pretrained(model, device=device)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            question = {"type": "noul", "instructions": row["instructions"]}
        else:
            question = {"type": row["type"], "instructions": row["instructions"],
                        "options": [text for _, text in pairs]}
        out = self.model.decide(state_text(row["state"]), [question])[0]
        if row["type"] == "noul":
            pred = "yes" if out["noul"] >= 0.5 else "no"
            probs = {"no": 1.0 - out["noul"], "yes": out["noul"]}
        else:
            by_text = out["probabilities"]
            ordered = [by_text[text] for _, text in pairs]
            probs = {idx: p for (idx, _), p in zip(pairs, ordered)}
            if row["type"] == "choice":
                pred = pairs[max(range(len(ordered)), key=ordered.__getitem__)][0]
            else:
                pred = max(range(len(ordered)), key=ordered.__getitem__)
        return pred, {"probabilities": probs, "adapter": "typed-decisions open_jev native"}


class DeciderAdapter:
    def __init__(self, model: str, device: str, revision: str | None):
        import torch
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model, revision=revision) if revision else model
        sys.path.insert(0, model_path)
        from decider.infer import Decider
        self.decider = Decider(model_path, device=device, dtype=torch.float16,
                               use_graphs=False)

    def predict(self, row: dict) -> tuple[str, dict]:
        question = {"type": row["type"], "instructions": row["instructions"]}
        if row["criteria"] is not None:
            question["criteria"] = row["criteria"]
        out = self.decider.system_one(
            state_text(row["state"]), {"q": question}, independent=True)["answers"]["q"]
        if row["type"] == "noul":
            pred = "yes" if out["noul"] >= 0.5 else "no"
            probs = {"no": 1.0 - out["noul"], "yes": out["noul"]}
        else:
            probs = {str(k): float(v) for k, v in out["probabilities"].items()}
            pred = max(probs, key=probs.get)
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "Mapika decider.infer system_one independent=True"}


class OpenAlternativeAdapter:
    def __init__(self, model: str, device: str, revision: str | None):
        sys.path.insert(0, str(ROOT / "external/open-alternative-jev"))
        import torch
        from huggingface_hub import snapshot_download
        from so1 import Choice, Decider
        import os
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        from transformers import AutoModelForImageTextToText
        model_path = snapshot_download(model, revision=revision) if revision else model
        self.Choice = Choice
        self.decider = Decider.from_pretrained(
            model_path, backend="hf", dtype=torch.float16,
            device_map=device, model_class=AutoModelForImageTextToText,
            mode="separate")

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            options = ["yes", "no"]
        else:
            options = [text for _, text in pairs]
        decision = self.decider.decide(
            state_text(row["state"]), [self.Choice(row["instructions"], options)],
            mode="separate")[0]
        if row["type"] == "noul":
            probs = {"yes": float(decision.probabilities[0]),
                     "no": float(decision.probabilities[1])}
            pred = "yes" if probs["yes"] >= probs["no"] else "no"
        else:
            probs = {idx: float(p) for (idx, _), p in zip(pairs, decision.probabilities)}
            pred = pairs[decision.index][0]
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "open-alternative-jev HFBackend mode=separate"}


class ReflexAdapter:
    def __init__(self, model: str, device: str, revision: str | None):
        sys.path.insert(0, str(ROOT / "external/reflex/src"))
        import torch
        from huggingface_hub import snapshot_download
        from reflex import Engine, SystemOneRequest
        model_path = snapshot_download(model, revision=revision) if revision else model
        self.SystemOneRequest = SystemOneRequest
        self.engine = Engine.load(model_path, dtype=torch.float16, device=device,
                                  prompt_style="markdown", default_permutations=2)

    def predict(self, row: dict) -> tuple[str, dict]:
        question = {"type": row["type"], "instructions": row["instructions"]}
        if row["criteria"] is not None:
            question["criteria"] = row["criteria"]
        out = self.engine.answer(self.SystemOneRequest(
            state=state_text(row["state"]), questions={"q": question})).answers["q"]
        if row["type"] == "noul":
            pred = "yes" if out.noul >= 0.5 else "no"
            probs = {"no": 1.0 - out.noul, "yes": out.noul}
        else:
            probs = {str(k): float(v) for k, v in out.probabilities.items()}
            pred = max(probs, key=probs.get)
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "reflex Engine stable config; frozen model; 2 permutations"}


class SemIfAdapter:
    def __init__(self, model: str, device: str, revision: str, mlx_bits: int | None):
        sys.path.insert(0, str(ROOT / "external/semif/src"))
        from semif_phase1 import mlx_backend
        self.mlx_backend = mlx_backend
        self.model, self.tokenizer, self.metadata = mlx_backend.load_model(
            model, revision, mlx_bits)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            options = [
                {"id": "no", "description": "The proposition is false."},
                {"id": "yes", "description": "The proposition is true."},
            ]
        else:
            options = [{"id": idx, "description": text} for idx, text in pairs]
        out = self.mlx_backend.score(
            self.model, self.tokenizer,
            {"id": row["id"], "state": state_text(row["state"]),
             "question": row["instructions"], "options": options},
            self.metadata)
        ordered = [float(p) for p in out["probabilities"]]
        probs = {option["id"]: p for option, p in zip(options, ordered)}
        pred = options[max(range(len(ordered)), key=ordered.__getitem__)]["id"]
        if row["type"] == "score":
            pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "semif_phase1 mlx_backend.score direct native",
                      "input_tokens": out["input_tokens"]}


class CertoAdapter:
    def __init__(self, model: str, device: str):
        sys.path.insert(0, str(ROOT / "external/certo"))
        from huggingface_hub import snapshot_download
        from infer import DecisionModel
        self.model = DecisionModel.load(snapshot_download(model), device=device)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            options = [{"id": "no", "description": "false or unsupported"},
                       {"id": "yes", "description": "true or supported"}]
        else:
            options = [{"id": idx, "description": text} for idx, text in pairs]
        out = self.model.decide(state_text(row["state"]), options)
        probs = {str(k): float(v) for k, v in out["probs"].items()}
        pred = out["top"]
        if row["type"] == "score":
            pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "certo DecisionModel.decide; state truncated to 64 tokens by upstream default"}


class SmallJevAdapter:
    def __init__(self, model: str, device: str, adapter_id: str | None,
                 sem_ckpt: str | None, heads_ckpt: str | None):
        sys.path.insert(0, str(ROOT / "external/smalljev"))
        import torch
        from smalljev.model import HFBackend
        from smalljev.spec import ChoiceQuestion, NoulQuestion, ScoreQuestion
        self.ChoiceQuestion = ChoiceQuestion
        self.NoulQuestion = NoulQuestion
        self.ScoreQuestion = ScoreQuestion
        self.backend = HFBackend(model_id=model, adapter_id=adapter_id,
                                 heads_ckpt=heads_ckpt, sem_ckpt=sem_ckpt)
        if device == "mps" and torch.backends.mps.is_available():
            self.backend.model.to("mps")
            self.backend.device = torch.device("mps")
            if self.backend.heads is not None:
                self.backend.heads.to("mps")
            if self.backend.sem_head is not None:
                self.backend.sem_head.to("mps")

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            q = self.NoulQuestion("q", row["instructions"])
        elif row["type"] == "score":
            q = self.ScoreQuestion("q", row["instructions"],
                                   tuple(text for _, text in pairs),
                                   tuple(float(i) for i, _ in enumerate(pairs)))
        else:
            q = self.ChoiceQuestion("q", row["instructions"],
                                    tuple(text for _, text in pairs))
        out = self.backend.score_all_semantic(state_text(row["state"]), {"q": q})["q"]
        if row["type"] == "noul":
            pred = "yes" if float(out) >= 0.5 else "no"
            probs = {"no": 1.0 - float(out), "yes": float(out)}
        else:
            ordered = [float(p) for p in out]
            probs = {idx: p for (idx, _), p in zip(pairs, ordered)}
            pred = pairs[max(range(len(ordered)), key=ordered.__getitem__)][0]
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "smalljev semantic-v9 native; forced MPS device after load"}


class LayaAdapter:
    def __init__(self, model: str, device: str, subfolder: str | None = None):
        import laya
        self.model = laya.load(model, device=device, subfolder=subfolder)

    def predict(self, row: dict) -> tuple[str, dict]:
        question = {"type": row["type"], "instructions": row["instructions"]}
        if row["criteria"] is not None:
            question["criteria"] = row["criteria"]
        out = self.model.predict(state_text(row["state"]), {"q": question})["answers"]["q"]
        if row["type"] == "noul":
            pred = "yes" if out["noul"] >= 0.5 else "no"
            probs = {"no": 1.0 - out["noul"], "yes": out["noul"]}
        elif row["type"] == "score":
            probs = {str(k): float(v) for k, v in out["probabilities"].items()}
            pred = int(max(probs, key=probs.get))
        else:
            probs = {str(k): float(v) for k, v in out["probabilities"].items()}
            pred = max(probs, key=probs.get)
        return pred, {"probabilities": probs, "adapter": "laya native system_one"}


class GavelAdapter:
    def __init__(self, model: str, device: str):
        sys.path.insert(0, str(ROOT / "external/gavel/src"))
        from gavel import Gavel
        self.model = Gavel.from_pretrained(model, device=device)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            pairs = [("no", pairs[0][1] if pairs else "false"),
                     ("yes", pairs[1][1] if len(pairs) > 1 else "true")]
            # Gavel criteria ordering is usually false,true; keep explicit ids.
            pairs = [("no", "false: " + pairs[0][1]),
                     ("yes", "true: " + pairs[1][1])]
        result = self.model.decide(state_text(row["state"]), row["instructions"],
                                   [text for _, text in pairs])
        probs = {idx: result.probabilities[text] for idx, text in pairs}
        pred = next(idx for idx, text in pairs if text == result.option)
        if row["type"] == "score" and pred.isdigit():
            pred = int(pred)
        return pred, {"probabilities": probs, "adapter": "gavel pair-wise entailment; noul/score mapped to options"}


class OpenJevZefanAdapter:
    def __init__(self, model: str, device: str, revision: str | None,
                 base_model: str | None):
        import torch
        from huggingface_hub import snapshot_download
        checkpoint_root = snapshot_download(model, revision=revision) if revision else model
        checkpoint = Path(checkpoint_root) / "package/checkpoint"
        source_root = snapshot_download(
            "ZefanCai/Open-Jev", repo_type="dataset",
            allow_patterns=["reproduce/source-code/**"])
        sys.path.insert(0, str(Path(source_root) / "reproduce/source-code"))
        from jev.api import compile_request, format_response
        from jev.metrics import softmax
        from jev.model import DecisionModel
        self.compile_request = compile_request
        self.format_response = format_response
        self.softmax = softmax
        self.torch = torch
        self.temperature = json.loads((checkpoint / "temperature.json").read_text())["temperature"]
        if base_model:
            config = json.loads((checkpoint / "model.json").read_text())
            self.model = DecisionModel(base_model, None, device="cpu",
                                       lora_rank=0, max_length=config["max_length"])
            if config["lora_rank"]:
                from peft import PeftModel
                self.model.backbone = PeftModel.from_pretrained(
                    self.model.backbone, checkpoint / "adapter")
                self.model.lora_rank = config["lora_rank"]
            self.model.head.load_state_dict(torch.load(
                checkpoint / "head.pt", map_location="cpu", weights_only=True))
            self.model.eval()
        else:
            self.model = DecisionModel.load(checkpoint, device="cpu")
        if device == "mps":
            self.model = self.model.to("mps")
            self.model.device_name = "mps"

    def predict(self, row: dict) -> tuple[str, dict]:
        question = {"type": row["type"], "instructions": row["instructions"]}
        if row["criteria"] is not None:
            question["criteria"] = row["criteria"]
        records = self.compile_request(state_text(row["state"]), {"q": question})
        try:
            with self.torch.inference_mode():
                logits = self.model(records)
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
            return abstain_result(row, str(exc))
        probs = [self.softmax(item.float().cpu().tolist(), temperature=self.temperature)
                 for item in logits]
        out = self.format_response(records, probs)["answers"]["q"]
        if row["type"] == "noul":
            p_yes = float(out["noul"])
            pred = "yes" if p_yes >= 0.5 else "no"
            answer_probs = {"no": 1.0 - p_yes, "yes": p_yes}
        else:
            answer_probs = {str(k): float(v) for k, v in out["probabilities"].items()}
            pred = max(answer_probs, key=answer_probs.get)
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": answer_probs,
                      "adapter": "Zefan Open-Jev native independent candidates; saved temperature",
                      "input_tokens": self.model.last_input_tokens}


class AgentJevAdapter:
    def __init__(self, checkpoint: str, model_path: str, device: str,
                 temperatures: str | None):
        sys.path.insert(0, str(ROOT / "external/agent-jev"))
        from jev_service.engine import DecisionEngine
        self.engine = DecisionEngine(checkpoint, model_path, device=device,
                                     temperatures=temperatures)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            criteria = row["criteria"] if isinstance(row["criteria"], dict) else {}
            question = {"id": "q", "type": "boolean", "question": row["instructions"],
                        "criteria": {"true": criteria.get("true", "TRUE"),
                                     "false": criteria.get("false", "FALSE")}}
        elif row["type"] == "score":
            question = {"id": "q", "type": "score", "question": row["instructions"],
                        "levels": [text for _, text in pairs]}
        else:
            question = {"id": "q", "type": "choice", "question": row["instructions"],
                        "options": dict(pairs)}
        try:
            response = self.engine.evaluate({"state": row["state"], "questions": [question]})
        except ValueError as exc:
            return abstain_result(row, str(exc))
        answer = response["results"][0]["answers"][0]
        usage = response.get("usage", {})
        if row["type"] == "noul":
            probs = {"no": float(answer["distribution"]["false"]),
                     "yes": float(answer["distribution"]["true"])}
            pred = "yes" if probs["yes"] >= probs["no"] else "no"
        else:
            probs = {idx: float(p) for (idx, _), p in zip(
                pairs, answer["distribution"].values())}
            if row["type"] == "score":
                pred = int(answer["level"])
            else:
                pred = str(answer["value"])
        return pred, {"probabilities": probs,
                      "adapter": "AgentJev native DecisionEngine",
                      "input_tokens": usage.get("input_path_tokens"),
                      "backbone_input_tokens": usage.get("backbone_input_tokens"),
                      "encoder": response.get("usage", {}).get("encoder")}


class ThisThatAdapter:
    def __init__(self, model: str, device: str, max_state_tokens: int | None):
        sys.path.insert(0, str(ROOT / "external/this-that-model"))
        from thisthat import Question, TypedDecider
        from thisthat.prompt import DEFAULT_MAX_STATE_TOKENS, build
        self.Question = Question
        self.build = build
        self.max_state_tokens = max_state_tokens or DEFAULT_MAX_STATE_TOKENS
        self.decider = TypedDecider.from_pretrained(model, device=device)

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            criteria = row["criteria"] if isinstance(row["criteria"], dict) else {}
            options = [str(criteria.get("false", "no")),
                       str(criteria.get("true", "yes"))]
        else:
            options = [text for _, text in pairs]
        try:
            question = self.Question(row["instructions"], options)
            built = self.build(self.decider.tokenizer, state_text(row["state"]),
                               [question], max_state_tokens=self.max_state_tokens)
            decision = self.decider.decide(state_text(row["state"]), question,
                                           max_state_tokens=self.max_state_tokens)
        except ValueError as exc:
            return abstain_result(row, str(exc))
        if row["type"] == "noul":
            probs = {"no": float(decision.probabilities[0]),
                     "yes": float(decision.probabilities[1])}
            pred = "yes" if probs["yes"] >= probs["no"] else "no"
        else:
            probs = {idx: float(p) for (idx, _), p in zip(
                pairs, decision.probabilities)}
            pred = pairs[decision.index][0]
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "this-that TypedDecider native one-pass option-label head",
                      "input_tokens": len(built["ids"]),
                      "max_state_tokens": self.max_state_tokens}


class JevLocalAdapter:
    def __init__(self, model: str, device: str):
        sys.path.insert(0, str(ROOT / "external/jev-local/src"))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from jevlocal.models import ChoiceQuestion, NoulQuestion, ScoreQuestion
        from jevlocal.scorer import HfLogprobScorer, default_temperatures
        self.ChoiceQuestion = ChoiceQuestion
        self.NoulQuestion = NoulQuestion
        self.ScoreQuestion = ScoreQuestion
        self.scorer = HfLogprobScorer.__new__(HfLogprobScorer)
        self.scorer.model_name = "Qwen/Qwen3.5-9B"
        self.scorer.chat = False
        self.scorer.temperatures = default_temperatures("Qwen/Qwen3.5-9B", False)
        self.scorer.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.scorer.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16, trust_remote_code=True)
        self.scorer.model.to(device).eval()

    def predict(self, row: dict) -> tuple[str, dict]:
        state = row["state"] if isinstance(row["state"], str) else str(row["state"])
        if row["type"] == "noul":
            q = self.NoulQuestion(type="noul", instructions=row["instructions"],
                                criteria=row.get("criteria"))
            answer = self.scorer.noul(state, q)
            probs = {"no": 1.0 - answer.noul, "yes": answer.noul}
            pred = "yes" if answer.noul >= 0.5 else "no"
            out = {"type": "noul", "noul": answer.noul, "probabilities": probs}
        elif row["type"] == "choice":
            q = self.ChoiceQuestion(type="choice", instructions=row["instructions"],
                                    criteria=row["criteria"])
            answer = self.scorer.choice(state, q)
            probs = {k: float(v) for k, v in answer.probabilities.items()}
            pred = answer.choice
            out = {"type": "choice", "choice": pred, "probabilities": probs,
                   "confidence": answer.confidence}
        else:
            q = self.ScoreQuestion(type="score", instructions=row["instructions"],
                                   criteria=row["criteria"])
            answer = self.scorer.score(state, q)
            probs = {int(k): float(v) for k, v in answer.probabilities.items()}
            pred = int(max(probs, key=probs.get))
            out = {"type": "score", "score": answer.score, "probabilities": probs,
                   "confidence": answer.confidence, "legend": answer.legend}
        out["adapter"] = "jev-local HfLogprobScorer native candidate mean-logprob"
        out["temperatures"] = dict(self.scorer.temperatures)
        return pred, out


class NimbleAdapter:
    def __init__(self, model: str, device: str, revision: str | None,
                 base_model: str | None):
        import hashlib
        import torch
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model, revision=revision) if revision else model
        model_dir = Path(model_path)
        sys.path.insert(0, str(model_dir))
        from parallel_schema import prepare_prompts
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.prepare_prompts = prepare_prompts
        self.torch = torch
        self.contract = json.loads((model_dir / "schema_config.json").read_text())
        assert self.contract["task"] == "schema_candidate_classification_v1"
        assert hashlib.sha256((model_dir / "parallel_schema.py").read_bytes()).hexdigest() == self.contract["prompt_code_sha256"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_path = base_model or self.contract["model"]
        base_revision = None if base_model else self.contract["revision"]
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_path, revision=base_revision,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        base.config.use_cache = False
        self.model = PeftModel.from_pretrained(base, model_dir).eval()
        self.device = device
        self.pad_id = self.tokenizer.pad_token_id

    def predict(self, row: dict) -> tuple[str, dict]:
        pairs = criteria_pairs(row["criteria"])
        if row["type"] == "noul":
            field = {"type": "boolean", "description": row["instructions"]}
        elif row["type"] == "score":
            field = {"type": "enum", "description": row["instructions"],
                     "choices": [str(i) for i, _ in pairs],
                     "choice_descriptions": {str(i): text for i, text in pairs}}
        else:
            field = {"type": "enum", "description": row["instructions"],
                     "choices": [idx for idx, _ in pairs],
                     "choice_descriptions": dict(pairs)}
        try:
            prepared = self.prepare_prompts(
                self.tokenizer, state_text(row["state"]), {"q": field},
                self.contract["max_length"])
        except ValueError as exc:
            return abstain_result(row, str(exc))
        ids = prepared.full_ids[0]
        candidate_ids = prepared.candidate_ids[0]
        inputs = {
            "input_ids": self.torch.tensor([ids], device=self.device),
            "attention_mask": self.torch.ones(1, len(ids), dtype=self.torch.long, device=self.device),
        }
        with self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        selected = logits[0, self.torch.tensor(candidate_ids, device=self.device)].float().cpu()
        probs_list = selected.softmax(-1).tolist()
        if row["type"] == "noul":
            probs = {"no": float(probs_list[0]), "yes": float(probs_list[1])}
            pred = "yes" if probs["yes"] >= probs["no"] else "no"
        else:
            probs = {idx: float(p) for (idx, _), p in zip(pairs, probs_list)}
            pred = max(probs, key=probs.get)
            if row["type"] == "score":
                pred = int(pred)
        return pred, {"probabilities": probs,
                      "adapter": "Bespoke Nimble reference candidate-token scoring on MPS",
                      "input_tokens": len(ids)}


def make_adapter(args: argparse.Namespace):
    if args.engine == "verdict":
        return VerdictAdapter(args.model, args.device)
    if args.engine == "openjev-deberta":
        return OpenJevAdapter(args.model, args.device)
    if args.engine == "gavel":
        return GavelAdapter(args.model, args.device)
    if args.engine == "laya":
        return LayaAdapter(args.model, args.device, args.subfolder)
    if args.engine == "smalljev":
        return SmallJevAdapter(args.model, args.device, args.adapter_id,
                               args.sem_ckpt, args.heads_ckpt)
    if args.engine == "certo":
        return CertoAdapter(args.model, args.device)
    if args.engine == "semif":
        return SemIfAdapter(args.model, args.device, args.revision, args.mlx_bits)
    if args.engine == "reflex":
        return ReflexAdapter(args.model, args.device, args.revision)
    if args.engine == "open-alternative":
        return OpenAlternativeAdapter(args.model, args.device, args.revision)
    if args.engine == "decider":
        return DeciderAdapter(args.model, args.device, args.revision)
    if args.engine == "open-jev-zefan":
        return OpenJevZefanAdapter(args.model, args.device, args.revision,
                                   args.base_model)
    if args.engine == "agentjev":
        return AgentJevAdapter(args.model, args.base_model, args.device,
                               args.temperatures)
    if args.engine == "thisthat":
        return ThisThatAdapter(args.model, args.device, args.max_state_tokens)
    if args.engine == "jevlocal":
        return JevLocalAdapter(args.model, args.device)
    if args.engine == "nimble":
        return NimbleAdapter(args.model, args.device, args.revision,
                             args.base_model)
    raise ValueError(args.engine)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True,
                        choices=["verdict", "openjev-deberta", "gavel", "laya", "smalljev", "certo", "semif", "reflex", "open-alternative", "decider", "open-jev-zefan", "agentjev", "thisthat", "jevlocal", "nimble"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--temperatures")
    parser.add_argument("--max-state-tokens", type=int)
    parser.add_argument("--subfolder")
    parser.add_argument("--adapter-id")
    parser.add_argument("--sem-ckpt")
    parser.add_argument("--heads-ckpt")
    parser.add_argument("--revision")
    parser.add_argument("--mlx-bits", type=int, choices=[4, 8])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tracks")
    args = parser.parse_args()

    rows = load_rows()
    wanted = set(args.tracks.split(",")) if args.tracks else None
    selected = [r for r in rows if wanted is None or r["track"] in wanted]
    done = set()
    if args.receipt.exists():
        for line in args.receipt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add((rec["track"], rec["id"]))
    pending = [r for r in selected if (r["track"], r["id"]) not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(json.dumps({"rows_total": len(rows), "selected": len(selected),
                      "already_done": len(done), "pending": len(pending),
                      "engine": args.engine, "model": args.model}, indent=2))
    if not pending:
        return 0

    adapter = make_adapter(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                pred, answer = adapter.predict(row)
                rec = {
                    "track": row["track"], "id": row["id"], "family": row["family"],
                    "gold": row["gold"], "pred": pred,
                    "correct": None if row["gold"] is None else pred == row["gold"],
                    "answer": answer, "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "model": args.model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            except Exception as exc:
                rec = {"track": row["track"], "id": row["id"],
                       "error": {"name": type(exc).__name__, "message": str(exc)}}
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(pending):
                print(json.dumps({"completed": index, "of": len(pending)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
