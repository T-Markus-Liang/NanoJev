import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_offline_jevbench_bundle_v1 as bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_offline_jevbench_bundle_v1.py"
BUNDLE = ROOT / "data" / "jevbench_offline_bundle_v1"


def run(root, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
        capture_output=True,
        text=True,
    )


def make_bundle(root: Path) -> None:
    (root / "track_a").mkdir(parents=True)
    (root / "track_a" / "rows.jsonl").write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    (root / "track_a" / "notes.txt").write_text("hello", encoding="utf-8")
    files = [
        bundle.artifact(root, "track_a/rows.jsonl", format_name="jsonl"),
        bundle.artifact(root, "track_a/notes.txt", format_name="file"),
    ]
    (root / "offline_bundle_manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )


class SyntheticVerifyTests(unittest.TestCase):
    def test_verify_clean_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            self.assertEqual(run(root, "--verify").returncode, 0)

    def test_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            with (root / "track_a" / "rows.jsonl").open("a") as fh:
                fh.write('{"id": 3}\n')
            result = run(root, "--verify")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rows.jsonl", result.stdout)

    def test_verify_detects_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            (root / "track_a" / "notes.txt").unlink()
            self.assertNotEqual(run(root, "--verify").returncode, 0)

    def test_verify_fails_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "track_a").mkdir(parents=True)
            self.assertNotEqual(run(root, "--verify").returncode, 0)


@unittest.skipUnless(
    (BUNDLE / "offline_bundle_manifest.json").is_file(),
    "offline bundle not downloaded on this machine",
)
class RealBundleTests(unittest.TestCase):
    def test_real_bundle_verifies(self):
        result = run(BUNDLE, "--verify")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_manifest_marks_all_tracks_evaluation_only(self):
        manifest = json.loads((BUNDLE / "offline_bundle_manifest.json").read_text())
        for track in manifest["tracks"]:
            if track.get("status") == "blocked_missing_redistributable_snapshot":
                continue
            self.assertTrue(track["evaluation_only"], track["id"])
            self.assertFalse(track["training_allowed"], track["id"])
            self.assertFalse(track["calibration_fit_allowed"], track["id"])


if __name__ == "__main__":
    unittest.main()
