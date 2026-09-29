"""Offline tests for scripts/rewindow_for_winnow_v1.py — synthetic states,
no winnow server required (the measurer is injected)."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewindow_for_winnow_v1 as rw


def entry(i, role, content):
    return {"pointer": f"/messages/{i}/content", "role": role,
            "content": content}


def make_request(conversation, candidate_pointer, users):
    state = {"conversation": conversation,
             "candidate_pointer": candidate_pointer,
             "user_messages_in_order": users}
    return {"state": rw.serialized(state),
            "questions": {"irrelevant": {"type": "noul",
                                         "instructions": "irrelevant?"}}}


def content_tokens(request):
    """Fake winnow-prefix measure: one token per content char (deterministic,
    includes elided markers so tests exercise marker cost too)."""
    state = json.loads(request["state"])
    return sum(len(e["content"]) for e in state["conversation"])


def conversation_of(request):
    return json.loads(request["state"])["conversation"]


def pointers_of(request):
    return [e["pointer"] for e in conversation_of(request)]


class ProtectedSetTests(unittest.TestCase):
    def test_protects_candidate_final_user_and_control(self):
        conv = [entry(0, "user", "old question"),
                entry(1, "assistant", "candidate text"),
                {"pointer": "/elided/0", "role": "control",
                 "content": "<elided 4 earlier segments>"},
                entry(2, "tool", "tool output"),
                entry(3, "user", "final request")]
        protected = rw.protected_indices(conv, "/messages/1/content")
        self.assertEqual(protected, {1, 2, 4})
        # /elided/N sidecar pointers are control-role -> protected
        self.assertIn(2, protected)

    def test_unresolvable_candidate_pointer_fails_protection(self):
        conv = [entry(0, "user", "hi"), entry(1, "assistant", "x")]
        self.assertIsNone(rw.protected_indices(conv, "/messages/9/content"))


class RewindowTests(unittest.TestCase):
    def setUp(self):
        # 100-char segments; protected: candidate idx1, control idx3,
        # final user idx5. Removable oldest-first: idx0, idx2, idx4.
        self.conv = [entry(0, "user", "u" * 100),
                     entry(1, "assistant", "c" * 100),
                     entry(2, "tool", "t" * 100),
                     {"pointer": "/elided/0", "role": "control",
                      "content": "<elided 9 earlier segments>"},
                     entry(4, "assistant", "a" * 100),
                     entry(5, "user", "f" * 100)]
        self.users = ["u" * 100, "f" * 100]
        self.request = make_request(self.conv, "/messages/1/content",
                                    self.users)

    def test_already_fitting_request_is_untouched(self):
        request, info = rw.rewindow_request(self.request, content_tokens,
                                            target=10_000)
        self.assertIs(request, self.request)
        self.assertFalse(info["rewindowed"])
        self.assertEqual(info["dropped_pointers"], [])
        self.assertFalse(info["overflow"])
        self.assertEqual(info["measured_prefix_tokens"],
                         info["final_prefix_tokens"])

    def test_drops_oldest_non_protected_first_and_converges(self):
        request, info = rw.rewindow_request(self.request, content_tokens,
                                            target=430)
        # 526 -> drop idx0 (~452) -> drop idx2 (~378 <= 430): the newest
        # droppable idx4 survives, proving strictly oldest-first order.
        self.assertTrue(info["rewindowed"])
        self.assertEqual(info["dropped_pointers"],
                         ["/messages/0/content", "/messages/2/content"])
        self.assertLessEqual(info["final_prefix_tokens"], 430)
        self.assertFalse(info["overflow"])
        kept = pointers_of(request)
        self.assertIn("/messages/1/content", kept)   # candidate
        self.assertIn("/messages/5/content", kept)   # final user
        self.assertIn("/messages/4/content", kept)   # newest droppable kept
        self.assertIn("/elided/0", kept)             # pre-existing marker
        self.assertNotIn("/messages/0/content", kept)
        self.assertNotIn("/messages/2/content", kept)

    def test_new_elided_markers_renumber_after_existing(self):
        request, _ = rw.rewindow_request(self.request, content_tokens,
                                         target=430)
        elided = [e for e in conversation_of(request)
                  if e["pointer"].startswith("/elided/")]
        self.assertIn("/elided/0", [e["pointer"] for e in elided])
        new = [e for e in elided if e["pointer"] != "/elided/0"]
        self.assertTrue(new)
        for marker in new:
            self.assertGreater(int(marker["pointer"].rsplit("/", 1)[1]), 0)
            self.assertTrue(
                marker["content"].startswith("<elided ")
                and marker["content"].endswith(" earlier segments>"))
            self.assertEqual(marker["role"], "control")

    def test_sidecar_user_messages_preserved_verbatim(self):
        request, _ = rw.rewindow_request(self.request, content_tokens,
                                         target=100)
        state = json.loads(request["state"])
        self.assertEqual(state["user_messages_in_order"], self.users)
        self.assertEqual(state["candidate_pointer"], "/messages/1/content")

    def test_overflow_marks_flag_and_keeps_protected_floor(self):
        def floor_measure(request):  # never fits: floor above target
            return max(content_tokens(request), 5_000)

        request, info = rw.rewindow_request(self.request, floor_measure,
                                            target=400)
        self.assertTrue(info["overflow"])
        self.assertEqual(info["reason"], "protected_floor_exceeds_target")
        self.assertEqual(info["dropped_pointers"],
                         ["/messages/0/content", "/messages/2/content",
                          "/messages/4/content"])  # every removable tried
        kept = pointers_of(request)
        self.assertIn("/messages/1/content", kept)
        self.assertIn("/messages/5/content", kept)
        self.assertIn("/elided/0", kept)

    def test_candidate_is_never_dropped_even_when_oldest(self):
        conv = [entry(0, "assistant", "candidate"),  # candidate is oldest
                entry(1, "assistant", "x" * 100),
                entry(2, "user", "final question")]
        request = make_request(conv, "/messages/0/content", ["final question"])

        def huge(request):
            return 9_999

        out, info = rw.rewindow_request(request, huge, target=100)
        self.assertTrue(info["overflow"])
        self.assertIn("/messages/0/content", pointers_of(out))
        self.assertNotIn("/messages/0/content", info["dropped_pointers"])
        self.assertIn("/messages/2/content", pointers_of(out))  # final user

    def test_unresolvable_candidate_pointer_fails_open_untouched(self):
        conv = [entry(0, "assistant", "x" * 100), entry(1, "user", "q")]
        request = make_request(conv, "/messages/9/content", ["q"])
        out, info = rw.rewindow_request(request, content_tokens, target=10)
        self.assertIs(out, request)  # request object untouched
        self.assertTrue(info["overflow"])
        self.assertEqual(info["reason"], "candidate_pointer_not_in_conversation")
        self.assertEqual(info["dropped_pointers"], [])


class RecordPipelineTests(unittest.TestCase):
    def test_rewindow_records_meta_and_summary(self):
        small = {"meta": {"record_id": "r:small"},
                 "request": make_request([entry(0, "user", "hi")],
                                         "/messages/0/content", ["hi"])}
        big = {"meta": {"record_id": "r:big", "windowed": False},
               "request": make_request([entry(0, "tool", "x" * 500),
                                        entry(1, "assistant", "cand"),
                                        entry(2, "user", "q")],
                                       "/messages/1/content", ["q"])}
        out, summary = rw.rewindow_records([small, big], content_tokens,
                                           target=300)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["over_target"], 1)
        self.assertEqual(summary["rewindowed"], 1)
        self.assertEqual(summary["fit"], 2)
        self.assertEqual(summary["overflow"], 0)
        self.assertEqual([r["meta"]["record_id"] for r in out],
                         ["r:small", "r:big"])
        note = out[1]["meta"]["winnow_rewindow"]
        self.assertEqual(note["dropped_pointers"], ["/messages/0/content"])
        self.assertTrue(out[1]["meta"]["windowed"])
        self.assertNotIn("winnow_overflow", out[1]["meta"])
        # Untouched record still gets provenance, unchanged request.
        self.assertIs(out[0]["request"], small["request"])
        self.assertFalse(out[0]["meta"]["winnow_rewindow"]["rewindowed"])


if __name__ == "__main__":
    unittest.main()
