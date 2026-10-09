import unittest

import run_e2e_official_upstream_v1 as e2e


MESSAGES = [
    {"role": "system", "content": "rules"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "part zero"},
        {"type": "text", "text": "part one noise"}]},
    {"role": "assistant", "content": "old note"},
    {"role": "user", "content": "question?"},
]


class PointerHandlingTest(unittest.TestCase):
    def test_resolve_whole_and_part_pointers(self):
        self.assertEqual(
            e2e.resolve_pointer(MESSAGES, "/messages/2/content"), "old note")
        self.assertEqual(
            e2e.resolve_pointer(MESSAGES, "/messages/1/content/1/text"),
            "part one noise")
        self.assertIsNone(e2e.resolve_pointer(MESSAGES, "/messages/9/content"))
        self.assertIsNone(
            e2e.resolve_pointer(MESSAGES, "/messages/1/content"))  # list

    def test_apply_drops_part_pointer_blanks_only_that_part(self):
        filtered = e2e.apply_drops(MESSAGES, ["/messages/1/content/1/text"])
        # message retained, only the targeted part blanked
        self.assertEqual(len(filtered), 4)
        self.assertEqual(filtered[1]["content"][0]["text"], "part zero")
        self.assertEqual(filtered[1]["content"][1]["text"], "")
        # original untouched
        self.assertEqual(MESSAGES[1]["content"][1]["text"], "part one noise")

    def test_apply_drops_whole_message(self):
        filtered = e2e.apply_drops(MESSAGES, ["/messages/2/content"])
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("old note", [m["content"] for m in filtered])


if __name__ == "__main__":
    unittest.main()
