#!/usr/bin/env python3
"""Generate data/context_relevance_v3 — additional context-filtering records.

v3 is a NEW generator (the original context_relevance_v1 generator is not in this
repository). It emits records in the same source schema as
data/context_relevance_v2_seed20260919 so the valen nano builder can consume them
unchanged:

  {id, state_id, family_id, split,
   state: {conversation:[{pointer,role,content}], candidate_pointer, family,
           wire_format, scenario_id},
   questions: {irrelevant: {type, instructions}},
   gold, gold_probs, gold_probs_kind, gold_label_kind,
   metadata: {source_group_id, scenario_family, candidate_kind, variant,
              hard_negative, source, license, ...}}

Candidate-kind roster: all 14 kinds already present across the two v2 sources
(context_relevance_*: required_evidence, overlap_distractor, correction,
user_constraint, wrong_entity, tool_dependency, unrelated, stale_fact;
oracle_*: required_field, two_fields, latest_correction, limit_check,
wrong_field, superseded) plus two NEW hard-negative families:

  near_duplicate_evidence (drop/true): the candidate merely restates
      information already present elsewhere in the state with trivial surface
      differences. Under the v2 contract the current value is in the state, so
      the redundant restatement is certainly irrelevant.
  correction_confirmed (keep/false): the candidate documents a correction that
      was later re-confirmed, i.e. it carries the current operative value; it
      looks like a stale-superseded record but must be kept.

Label contract (identical to VALEN_NANO_V2_CONTRACT_FIX_V1.md): a stale /
superseded candidate is irrelevant=true ONLY when the superseding current value
is present elsewhere in the state; when the stale fact is the sole evidence the
label stays false (the question itself says "answer false when uncertainty
remains"). Every "drop" template that references an outdated value is therefore
paired with a request variant that explicitly contains the current value, and
every "sole evidence" variant uses a request that does not. The chosen variant is
recorded in metadata.variant and the labels are a pure function of state content.

Groups: each group emits the full 16-kind roster for one parametrized scenario
and shares one source_group_id, so the sha256(group_id) split rule can never
place two views of the same scenario in different splits (strict group
isolation; v3 lineage prefix "context_relevance_v3" is applied by the builder).

Usage: python3 scripts/gen_context_relevance_v3_v1.py [--seed N]
       [--groups-per-family N] [--out DIR]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "context_relevance_v3"
DEFAULT_SEED = 20261005
DEFAULT_GROUPS_PER_FAMILY = 122  # 6 families * 122 groups * 16 kinds = 11 712 records

SCHEMA = "nanojev-context-relevance-v3"
FAMILY_ID = "context_relevance_v3"
INSTRUCTIONS = ("Is the candidate context certainly irrelevant to fulfilling the "
                "current user request? Answer false when it is required evidence, "
                "a user constraint, a correction, a tool dependency, or when "
                "uncertainty remains.")
SYSTEM_TEXT = "Preserve current user constraints and required evidence."

FAMILIES = ("code", "order", "risk", "support", "robotics", "multilingual")

# Fixed-roster kinds (16). correction / overlap_distractor exist in two variants
# ("current_in_state" -> drop, "sole_evidence" -> keep) chosen per record.
KINDS = (
    "required_evidence", "required_field", "two_fields", "limit_check",
    "latest_correction", "user_constraint", "tool_dependency",
    "correction", "overlap_distractor", "stale_fact", "unrelated",
    "wrong_entity", "wrong_field", "superseded",
    "near_duplicate_evidence", "correction_confirmed",
)

# Kinds deliberately built to be confusable (hard negatives). The new families
# and the whole correction/superseded/lookalike cluster are flagged so the
# merged eval set stays discriminative (>=30% hard-negative required).
HARD_KINDS = frozenset({
    "correction", "overlap_distractor", "superseded", "latest_correction",
    "wrong_entity", "wrong_field",
    "near_duplicate_evidence", "correction_confirmed",
})

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
ZH_PLACES = ("上海", "杭州", "北京", "深圳", "成都", "广州")


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def group_digest(seed, family, gidx):
    return hashlib.sha256(f"context_relevance_v3:{seed}:{family}:{gidx}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-family scenario emitters.
#
# Each emitter receives a params dict `p` and returns (requests, candidates):
#   requests: {"plain": str, "withval": str} — "withval" explicitly contains the
#             current value that any stale candidate was superseded by.
#   candidates: {kind: spec} where spec = {
#       "text": str or {"sole_evidence": str, "current_in_state": str},
#       "label": bool or {"sole_evidence": bool, "current_in_state": bool},
#       "role": "assistant" | "tool"  (default assistant),
#       "request": "plain" | "withval" | "either"   (default "either"),
#       "lang": {"zh": ..., "en": ...} optional per-language override (multilingual),
#   }
# ---------------------------------------------------------------------------

def _two_variant(text_sole, text_cur):
    return {"sole_evidence": text_sole, "current_in_state": text_cur}


def emit_code(p):
    sid, sid2 = p["sid"], p["sid2"]
    req_plain = f"Update the {sid} service to the corrected port and keep the health endpoint unchanged."
    req_val = f"Update the {sid} service to use port {p['new']} and keep the health endpoint unchanged."
    filler = p["filler"]
    cand = {
        "required_evidence": {"text": f"The current service port is {p['new']}; health endpoint is /health.", "label": False},
        "required_field": {"text": f"Field port for {sid}: {p['new']}.", "label": False},
        "two_fields": {"text": f"Service {sid}: port={p['new']}, workers={p['workers']}.", "label": False},
        "limit_check": {"text": f"Allowed port ceiling for {sid} is {p['maxport']}; the request must stay below it.", "label": False},
        "latest_correction": {"text": f"Latest correction for {sid}: port is now {p['new']} (was {p['old']}); this is the newest change.", "label": False},
        "user_constraint": {"text": None, "label": False},  # filled with request text
        "tool_dependency": {"text": f"Tool result for {sid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": _two_variant(
                f"An earlier draft for {sid} used port {p['old']} before it was corrected to {p['new']}.",
                f"A stale note says the {sid} service once used port {p['old']}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": _two_variant(
                f"An archived config for {sid} listed port {p['old']}; no newer port is recorded.",
                f"An archived config for {sid} listed port {p['old']}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": f"An archived task used port {p['other']} for a different service.", "label": True},
        "wrong_entity": {"text": f"The unrelated {sid2} service uses port {p['new']}.", "label": True},
        "wrong_field": {"text": f"Service {sid} currently runs {p['workers']} workers.", "label": True},
        "superseded": {"text": f"Superseded config: {sid} used port {p['old']} until the current setting replaced it.",
                       "label": True, "request": "withval"},
        "near_duplicate_evidence": {"text": f"Restated request parameters: service {sid}, port {p['new']}.",
                                    "label": True, "request": "withval"},
        "correction_confirmed": {"text": f"Correction history for {sid}: changed port {p['old']} -> {p['new']}; the user later re-confirmed {p['new']}.",
                                 "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


def emit_order(p):
    oid, oid2 = p["sid"], p["sid2"]
    req_plain = f"For order {oid}, confirm the requested quantity and price against the current quote."
    req_val = f"For order {oid}, confirm whether {p['q']} units at {p['u']} are within the requested budget."
    filler = p["filler"]
    cand = {
        "required_evidence": {"text": f"Current order {oid}: quantity={p['q']}, unit_price={p['u']}, budget={p['b']}.", "label": False},
        "required_field": {"text": f"Field unit_price for {oid}: {p['u']}.", "label": False},
        "two_fields": {"text": f"Order {oid}: quantity={p['q']}, unit_price={p['u']}.", "label": False},
        "limit_check": {"text": f"The budget limit for {oid} is {p['b']}; compare the requested total against it.", "label": False},
        "latest_correction": {"text": f"Latest correction for {oid}: quantity is now {p['q']} (was {p['q_old']}).", "label": False},
        "user_constraint": {"text": None, "label": False},
        "tool_dependency": {"text": f"Tool result for {oid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": _two_variant(
                f"An old quote for {oid} used quantity={p['q_old']} before it was corrected to {p['q']}.",
                f"An old quote for {oid} used quantity={p['q_old']} and unit_price={p['u']}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": _two_variant(
                f"A previous cart for {oid} held quantity={p['q_old']}; no current quote is stored.",
                f"A previous cart for {oid} held quantity={p['q_old']} at unit_price={p['u']}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": f"An archived order from another customer contained {p['other']} units.", "label": True},
        "wrong_entity": {"text": f"Order {oid2} has the same unit price but a different quantity.", "label": True},
        "wrong_field": {"text": f"Order {oid} is assigned to warehouse {p['warehouse']}.", "label": True},
        "superseded": {"text": f"Superseded quote: {oid} was priced at quantity={p['q_old']} before the current quote.",
                       "label": True, "request": "withval"},
        "near_duplicate_evidence": {"text": f"Restated request parameters: order {oid}, {p['q']} units at {p['u']}.",
                                    "label": True, "request": "withval"},
        "correction_confirmed": {"text": f"Correction history for {oid}: quantity {p['q_old']} -> {p['q']}; the customer later re-confirmed {p['q']}.",
                                 "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


def emit_risk(p):
    aid, aid2 = p["sid"], p["sid2"]
    req_plain = f"Can account {aid} open a position of {p['pos']} units under its current limit?"
    req_val = f"Can account {aid} open a position of {p['pos']} units under its current limit of {p['lim']}?"
    filler = p["filler"]
    cand = {
        "required_evidence": {"text": f"Current account {aid} limit is {p['lim']} units; requested position is {p['pos']}.", "label": False},
        "required_field": {"text": f"Field position_limit for {aid}: {p['lim']}.", "label": False},
        "two_fields": {"text": f"Account {aid}: position_limit={p['lim']}, order_limit={p['olim']}.", "label": False},
        "limit_check": {"text": f"Current limit for {aid} is {p['lim']} units; check the requested position against it.", "label": False},
        "latest_correction": {"text": f"Latest correction for {aid}: the limit is now {p['lim']} (was {p['lim_old']}).", "label": False},
        "user_constraint": {"text": None, "label": False},
        "tool_dependency": {"text": f"Tool result for {aid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": _two_variant(
                f"A previous risk snapshot for {aid} had a limit of {p['lim_old']} units before correction to {p['lim']}.",
                f"A previous risk snapshot for {aid} had a limit of {p['lim_old']} units."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": _two_variant(
                f"A prior exposure report for {aid} listed a {p['lim_old']}-unit cap; no newer limit is recorded.",
                f"A prior exposure report for {aid} listed a {p['lim_old']}-unit cap."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": f"A historical account had a temporary limit of {p['other']} units.", "label": True},
        "wrong_entity": {"text": f"Account {aid2} has a limit matching the requested position.", "label": True},
        "wrong_field": {"text": f"Account {aid} has a credit score tier of {p['tier']}.", "label": True},
        "superseded": {"text": f"Superseded snapshot: {aid} limit was {p['lim_old']}, replaced by the current limit.",
                       "label": True, "request": "withval"},
        "near_duplicate_evidence": {"text": f"Restated request parameters: account {aid}, requested position {p['pos']} units under limit {p['lim']}.",
                                    "label": True, "request": "withval"},
        "correction_confirmed": {"text": f"Correction history for {aid}: limit {p['lim_old']} -> {p['lim']}; the risk officer later re-confirmed {p['lim']}.",
                                 "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


def emit_support(p):
    tid, tid2 = p["sid"], p["sid2"]
    req_plain = (f"Resolve ticket {tid}: preserve the customer's requested refund "
                 f"deadline and cite the current policy.")
    req_val = (f"Resolve ticket {tid}: the refund deadline is {p['new']}; "
               f"cite the current policy.")
    filler = p["filler"]
    cand = {
        "required_evidence": {"text": f"Ticket {tid} requests a refund by {p['new']}; current policy allows refunds within {p['window']} days.", "label": False},
        "required_field": {"text": f"Field refund_deadline for {tid}: {p['new']}.", "label": False},
        "two_fields": {"text": f"Ticket {tid}: refund deadline={p['new']}, policy window={p['window']} days.", "label": False},
        "limit_check": {"text": f"Policy allows refunds within {p['window']} days for {tid}; the requested deadline must fit.", "label": False},
        "latest_correction": {"text": f"Latest correction for {tid}: the deadline is now {p['new']} (was {p['old']}).", "label": False},
        "user_constraint": {"text": None, "label": False},
        "tool_dependency": {"text": f"Tool result for {tid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": _two_variant(
                f"Ticket {tid} previously requested a refund by {p['old']} before the customer corrected it to {p['new']}.",
                f"Ticket {tid} previously requested a refund by {p['old']}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": _two_variant(
                f"An earlier ticket note for {tid} mentioned a {p['old']} deadline; no corrected deadline is recorded.",
                f"An earlier ticket note for {tid} mentioned a {p['old']} deadline."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": "An old ticket asked about changing a delivery address.", "label": True},
        "wrong_entity": {"text": f"Ticket {tid2} also requests a refund but under a different policy.", "label": True},
        "wrong_field": {"text": f"Ticket {tid} was opened by agent {p['agent']} and routed to refunds.", "label": True},
        "superseded": {"text": f"Superseded request: refund by {p['old']} for {tid}, replaced by the current deadline.",
                       "label": True, "request": "withval"},
        "near_duplicate_evidence": {"text": f"Restated request parameters: ticket {tid}, refund deadline {p['new']}.",
                                    "label": True, "request": "withval"},
        "correction_confirmed": {"text": f"Correction history for {tid}: deadline {p['old']} -> {p['new']}; the customer re-confirmed {p['new']} on a follow-up call.",
                                 "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


def emit_robotics(p):
    mid, mid2 = p["sid"], p["sid2"]
    req_plain = (f"For robot mission {mid}, select the safe speed after the "
                 f"operator's latest correction.")
    req_val = (f"For robot mission {mid}, apply the corrected speed limit of "
               f"{p['new']} m/s.")
    filler = p["filler"]
    cand = {
        "required_evidence": {"text": f"Mission {mid}: operator correction says speed must be at most {p['new']} m/s; current map is unchanged.", "label": False},
        "required_field": {"text": f"Field speed_limit for {mid}: {p['new']} m/s.", "label": False},
        "two_fields": {"text": f"Mission {mid}: speed_limit={p['new']} m/s, distance_limit={p['dist']} m.", "label": False},
        "limit_check": {"text": f"Mission {mid} safe speed ceiling is {p['new']} m/s; compare the proposed speed to it.", "label": False},
        "latest_correction": {"text": f"Latest correction for {mid}: speed is now {p['new']} m/s (was {p['old']}).", "label": False},
        "user_constraint": {"text": None, "label": False},
        "tool_dependency": {"text": f"Tool result for {mid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": _two_variant(
                f"An earlier plan for {mid} proposed {p['old']} m/s; the operator corrected it to {p['new']} m/s.",
                f"An earlier plan for {mid} proposed {p['old']} m/s before the operator correction."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": _two_variant(
                f"A prior route plan listed {p['old']} m/s for {mid}; no corrected speed is recorded.",
                f"A prior route plan listed {p['old']} m/s for {mid}."),
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": f"A completed mission used a {p['other']} m/s speed limit in another room.", "label": True},
        "wrong_entity": {"text": f"Mission {mid2} uses the same room but a different operator limit.", "label": True},
        "wrong_field": {"text": f"Mission {mid} is assigned to room {p['room']}.", "label": True},
        "superseded": {"text": f"Superseded plan: {mid} speed {p['old']} m/s, replaced by the operator's current limit.",
                       "label": True, "request": "withval"},
        "near_duplicate_evidence": {"text": f"Restated request parameters: mission {mid}, speed limit {p['new']} m/s.",
                                    "label": True, "request": "withval"},
        "correction_confirmed": {"text": f"Correction history for {mid}: speed {p['old']} -> {p['new']} m/s; the operator re-confirmed {p['new']} in the final review.",
                                 "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


def emit_multilingual(p, lang):
    """Multilingual family. `lang` selects the request language ("zh" or "en");
    several candidate kinds carry an explicit English twin so zh/en mixed
    records are produced deterministically."""
    tid, tid2 = p["sid"], p["sid2"]
    t_new, t_old = p["t_new"], p["t_old"]
    pl_new, pl_old = p["pl_new"], p["pl_old"]
    if lang == "zh":
        req_plain = f"请根据当前记录处理任务 {tid}，保留用户最后确认的时间和地点。"
        req_val = f"请根据当前记录处理任务 {tid}：用户已确认时间为 {t_new}、地点为 {pl_new}。"
        suffix = " Keep the latest user constraint."
    else:
        req_plain = (f"Process task {tid} against current records; keep the "
                     f"user's last confirmed time and place.")
        req_val = (f"Process task {tid}: the user confirmed time {t_new} and "
                   f"place {pl_new}.")
        suffix = " 保留最新的用户约束。"
    filler = p["filler"]

    def zh_en(zh, en):
        return {"zh": zh, "en": en}

    cand = {
        "required_evidence": {
            "text": zh_en(f"当前记录 {tid}：用户最后确认时间为 {t_new}，地点为 {pl_new}；之前的草稿已被更正。",
                          f"Current record {tid}: last confirmed time {t_new}, place {pl_new}; the earlier draft was corrected."),
            "label": False},
        "required_field": {"text": f"任务 {tid} 的确认时间字段值为 {t_new}。", "label": False},
        "two_fields": {"text": f"任务 {tid}：时间={t_new}，地点={pl_new}。", "label": False},
        "limit_check": {"text": f"当前 SLA 要求 {tid} 在 {t_new} 前完成；按用户确认时间校验。", "label": False},
        "latest_correction": {
            "text": zh_en(f"{tid} 的最新更正：时间改为 {t_new}（原为 {t_old}），地点改为 {pl_new}（原为 {pl_old}）。",
                          f"Latest correction for {tid}: time is now {t_new} (was {t_old}), place is now {pl_new} (was {pl_old})."),
            "label": False},
        "user_constraint": {"text": None, "label": False, "suffix": suffix},
        "tool_dependency": {"text": f"Tool result for {tid}: verified current record.", "label": False, "role": "tool"},
        "correction": {
            "text": {"sole_evidence": zh_en(
                         f"任务 {tid} 的旧草稿写的是 {t_old} 和 {pl_old}，但用户后来更正为 {t_new} 和 {pl_new}。",
                         f"The old draft for task {tid} said {t_old} and {pl_old}, but the user later corrected it to {t_new} and {pl_new}."),
                     "current_in_state": zh_en(
                         f"任务 {tid} 的旧草稿写的是 {t_old} 和 {pl_old}。",
                         f"An old draft for task {tid} said {t_old} and {pl_old}.")},
            "label": {"sole_evidence": False, "current_in_state": True}},
        "overlap_distractor": {
            "text": {"sole_evidence": zh_en(
                         f"历史缓存中 {tid} 的记录为 {t_old}、{pl_old}；没有更新的确认。",
                         f"The cached record for {tid} says {t_old} and {pl_old}; no newer confirmation exists."),
                     "current_in_state": zh_en(
                         f"历史缓存中 {tid} 的记录为 {t_old}、{pl_old}。",
                         f"The cached record for {tid} says {t_old} and {pl_old}.")},
            "label": {"sole_evidence": False, "current_in_state": True}},
        "stale_fact": {"text": filler, "label": True},
        "unrelated": {"text": f"一份无关的历史天气记录提到 {p['pl_other']} 和 09:00。", "label": True},
        "wrong_entity": {"text": f"任务 {tid2} 使用相同地点词汇，但不是当前任务。", "label": True},
        "wrong_field": {"text": f"任务 {tid} 的优先级标记为 P{p['prio']}。", "label": True},
        "superseded": {
            "text": zh_en(f"已废弃记录：{tid} 曾为 {t_old}/{pl_old}，已被当前确认覆盖。",
                          f"Deprecated record: {tid} was {t_old}/{pl_old}, covered by the current confirmation."),
            "label": True, "request": "withval"},
        "near_duplicate_evidence": {
            "text": zh_en(f"复述请求参数：任务 {tid}，时间 {t_new}，地点 {pl_new}。",
                          f"Restated request parameters: task {tid}, time {t_new}, place {pl_new}."),
            "label": True, "request": "withval"},
        "correction_confirmed": {
            "text": zh_en(f"更正记录 {tid}：{t_old}/{pl_old} -> {t_new}/{pl_new}；用户随后再次确认了 {t_new}/{pl_new}。",
                          f"Correction log for {tid}: {t_old}/{pl_old} -> {t_new}/{pl_new}; the user re-confirmed {t_new}/{pl_new} afterwards."),
            "label": False, "request": "plain"},
    }
    return {"plain": req_plain, "withval": req_val}, filler, cand


EMITTERS = {
    "code": emit_code, "order": emit_order, "risk": emit_risk,
    "support": emit_support, "robotics": emit_robotics,
    "multilingual": emit_multilingual,
}

FILLERS = {
    "code": ["The previous branch was deleted after a documentation-only change.",
             "A retired CI job once pinned an old toolchain version."],
    "order": ["The shipping label was printed last month and is not the current order.",
              "A packing slip from a cancelled order is filed separately."],
    "risk": ["A market commentary paragraph contains no account limit.",
             "An onboarding form lists contact details but no risk limits."],
    "support": ["The customer's old phone number is not needed for the policy answer.",
                "A satisfaction survey score is stored under a different ticket."],
    "robotics": ["Battery voltage from yesterday is not the corrected speed limit.",
                 "A calibration photo from last week shows the old room layout."],
    "multilingual": ["旧系统日志的语言设置与当前任务无关。",
                     "去年的导出文件只包含界面翻译，不包含确认记录。"],
}


def family_params(family, gidx, rng):
    p = {"filler": rng.choice(FILLERS[family]),
         "sid": f"{family}-{gidx:04d}",
         "sid2": f"{family}-{gidx + 5000:04d}"}
    if family == "code":
        p.update(new=rng.randrange(8000, 8500), old=rng.randrange(7000, 7999),
                 other=rng.randrange(9000, 9999), workers=rng.randrange(2, 9),
                 maxport=rng.randrange(9000, 9999))
    elif family == "order":
        q = rng.randrange(2, 9)
        u = rng.randrange(15, 60)
        p.update(q=q, q_old=q + rng.randrange(1, 5), u=u,
                 b=u * q + rng.randrange(10, 80), other=rng.randrange(50, 120),
                 warehouse=rng.randrange(1, 12))
    elif family == "risk":
        pos = rng.randrange(20, 80)
        lim = pos + rng.randrange(5, 40)
        p.update(pos=pos, lim=lim, lim_old=lim + rng.randrange(5, 30),
                 olim=rng.randrange(100, 400), other=rng.randrange(5, 15),
                 tier=rng.randrange(1, 6))
    elif family == "support":
        days = list(WEEKDAYS)
        new = days.pop(rng.randrange(len(days)))
        old = days.pop(rng.randrange(len(days)))
        p.update(new=new, old=old, window=rng.randrange(7, 30),
                 agent=rng.randrange(10, 99))
    elif family == "robotics":
        new = rng.choice([0.3, 0.4, 0.5])
        p.update(new=new, old=rng.choice([0.6, 0.7, 0.8, 1.0]),
                 other=rng.choice([1.2, 1.5, 2.0]), dist=rng.randrange(5, 40),
                 room=rng.choice(["A", "B", "C"]))
    elif family == "multilingual":
        places = list(ZH_PLACES)
        pl_new = places.pop(rng.randrange(len(places)))
        pl_old = places.pop(rng.randrange(len(places)))
        p.update(t_new=f"{rng.randrange(9, 19)}:{rng.choice(['00', '15', '30', '45'])}",
                 t_old=f"{rng.randrange(7, 12)}:{rng.choice(['00', '15', '30', '45'])}",
                 pl_new=pl_new, pl_old=pl_old,
                 pl_other=places[rng.randrange(len(places))],
                 prio=rng.randrange(1, 5))
    return p


def resolve_text(spec, variant):
    """Resolve the sole_evidence/current_in_state variant of a spec. May still
    return a {"zh": ..., "en": ...} pair for the multilingual family."""
    text = spec["text"]
    if isinstance(text, dict) and ("sole_evidence" in text or "current_in_state" in text):
        text = text[variant]
    return text


def has_cjk(text):
    return any("一" <= ch <= "鿿" for ch in text)


def pick_lang(request_lang, has_both, rng):
    """Pick a candidate language for the multilingual family; ~35% of zh requests
    and ~50% of en requests flip to the other language when both exist."""
    if not has_both:
        return None
    if request_lang == "zh":
        return "en" if rng.random() < 0.35 else "zh"
    return "zh" if rng.random() < 0.5 else "en"


def emit_group(family, gidx, seed, rng):
    """Emit the full 16-kind roster for one scenario; all records share a group."""
    p = family_params(family, gidx, rng)
    group = group_digest(seed, family, gidx)
    request_lang = None
    if family == "multilingual":
        request_lang = "zh" if rng.random() < 0.7 else "en"
        requests, filler, cand = emit_multilingual(p, request_lang)
    else:
        requests, filler, cand = EMITTERS[family](p)
    # Per-group request-variant draw for the kinds that tolerate either request.
    either_prefers_val = rng.random() < 0.5
    records = []
    for kind in KINDS:
        spec = dict(cand[kind])
        if kind in ("correction", "overlap_distractor"):
            variant = "current_in_state" if rng.random() < 0.5 else "sole_evidence"
            request_key = "withval" if variant == "current_in_state" else "plain"
            label = spec["label"][variant]
        else:
            variant = "canonical"
            label = spec["label"]
            request_key = spec.get("request")
            if request_key not in ("plain", "withval"):
                request_key = "withval" if either_prefers_val else "plain"
        text = resolve_text(spec, variant)
        if isinstance(text, dict):  # {"zh": ..., "en": ...} multilingual pair
            lang_choice = pick_lang(request_lang, True, rng)
            text = text.get(lang_choice) or text["zh"]
        lang_of_candidate = None
        if family == "multilingual" and kind != "user_constraint":
            lang_of_candidate = "zh" if has_cjk(text) else "en"
        if kind == "user_constraint":
            suffix = spec.get("suffix", " Keep the latest user constraint.")
            text = requests[request_key] + suffix
            if family == "multilingual":
                # zh request + en suffix (and vice versa) is a mixed record.
                lang_of_candidate = "en" if has_cjk(suffix) is False else "zh"
        records.append({
            "kind": kind, "variant": variant, "label": label,
            "request_key": request_key, "text": text,
            "role": spec.get("role", "assistant"),
            "hard": kind in HARD_KINDS,
            "mixed": (request_lang is not None and lang_of_candidate is not None
                      and lang_of_candidate != request_lang),
        })
    return group, requests, filler, records


def make_record(family, gidx, split, group, requests, filler, item):
    rid = f"context_v3:{split}:{family}:{gidx}:{item['kind']}"
    state = {
        "conversation": [
            {"pointer": "system", "role": "system", "content": SYSTEM_TEXT},
            {"pointer": "user", "role": "user", "content": requests[item["request_key"]]},
            {"pointer": "history", "role": "assistant", "content": filler},
            {"pointer": "candidate", "role": item["role"], "content": item["text"]},
        ],
        "candidate_pointer": "candidate",
        "family": family,
        "wire_format": "synthetic_canonical",
        "scenario_id": f"{family}-{gidx:04d}",
    }
    label = item["label"]
    metadata = {
        "source_group_id": group,
        "scenario_family": family,
        "candidate_kind": item["kind"],
        "variant": item["variant"],
        "hard_negative": item["hard"],
        "source": "self_authored_context_relevance_v3",
        "license": "CC0-1.0",
    }
    if item["mixed"]:
        metadata["mixed_language"] = True
    return {
        "id": rid, "state_id": rid, "family_id": FAMILY_ID, "split": split,
        "state": state,
        "questions": {"irrelevant": {"type": "boolean", "instructions": INSTRUCTIONS}},
        "gold": {"irrelevant": label},
        "gold_probs": {"irrelevant": {"false": float(not label), "true": float(label)}},
        "gold_probs_kind": {"irrelevant": "deterministic_truth"},
        "gold_label_kind": {"irrelevant": "deterministic_truth"},
        "metadata": metadata,
    }


def split_for(global_group_idx):
    """~5:1:1 rotation across groups so every split sees all families/kinds."""
    r = global_group_idx % 7
    return "train" if r < 5 else ("dev" if r == 5 else "calibration")


def generate(seed=DEFAULT_SEED, groups_per_family=DEFAULT_GROUPS_PER_FAMILY, out=DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "calibration": []}
    g = 0
    for family in FAMILIES:  # fixed order -> deterministic
        for gidx in range(groups_per_family):
            split = split_for(g)
            g += 1
            group, requests, filler, items = emit_group(family, gidx, seed, rng)
            for item in items:
                splits[split].append(
                    make_record(family, gidx, split, group, requests, filler, item))
    manifest = {
        "schema_version": SCHEMA,
        "seed": seed,
        "generator": "scripts/gen_context_relevance_v3_v1.py",
        "source": "self_authored_programmatic",
        "license": "CC0-1.0",
        "contract": ("stale/superseded candidate is drop(true) only when the superseding "
                     "current value is present elsewhere in the state; sole-evidence stale "
                     "facts stay keep(false); labels are a pure function of state content"),
        "kinds": list(KINDS),
        "hard_negative_kinds": sorted(HARD_KINDS),
        "new_families": ["near_duplicate_evidence", "correction_confirmed"],
        "limitations": [
            "Synthetic relevance labels; not production conversations.",
            "No frozen test/ood splits are emitted; v3 is consumed through train/dev/calibration only.",
            "Templates are finite; normalized near-duplicates exist by design and share labels.",
        ],
        "splits": {},
    }
    for split, rows in splits.items():
        text = "".join(serialized(r) + "\n" for r in rows)
        (out / f"{split}.jsonl").write_text(text, encoding="utf-8")
        families = Counter(r["metadata"]["scenario_family"] for r in rows)
        kinds = Counter(r["metadata"]["candidate_kind"] for r in rows)
        variants = Counter(r["metadata"]["variant"] for r in rows)
        hard = sum(1 for r in rows if r["metadata"]["hard_negative"])
        mixed = sum(1 for r in rows if r["metadata"].get("mixed_language"))
        labels = Counter(str(r["gold"]["irrelevant"]).lower() for r in rows)
        manifest["splits"][split] = {
            "records": len(rows),
            "questions": len(rows),
            "families": dict(families),
            "kinds": dict(kinds),
            "variants": dict(variants),
            "labels": dict(labels),
            "hard_negative": hard,
            "mixed_language": mixed,
            "source_groups": len({r["metadata"]["source_group_id"] for r in rows}),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: manifest["splits"][s]["records"] for s in splits}))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--groups-per-family", type=int, default=DEFAULT_GROUPS_PER_FAMILY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    generate(seed=args.seed, groups_per_family=args.groups_per_family, out=args.out)


if __name__ == "__main__":
    main()
