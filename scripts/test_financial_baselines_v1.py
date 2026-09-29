import copy
import json
from pathlib import Path
import tempfile
import unittest

import financial_baselines_v1 as fb

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'results/financial_pit_r1_bound_receipt_20260920_v5.json'


class BaselineDataTests(unittest.TestCase):
    def test_real_bound_cohort_recomputed(self):
        _, rows, folds = fb.load_bound_cohort(RECEIPT)
        self.assertEqual(len(rows), 3266)
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(f['all_phases_nonempty'] for f in folds))

    def test_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x.json'
            p.write_text('{}')
            ref = fb.identity(p)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                fb.verified_artifact(ref)

    def test_receipt_fold_tamper_rejected(self):
        receipt = json.loads(RECEIPT.read_text())
        receipt['folds'][0]['retained']['dev'].pop()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'receipt.json'
            p.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'recomputed PIT folds'):
                fb.load_bound_cohort(p)

    def test_failed_binding_rejected(self):
        receipt = json.loads(RECEIPT.read_text())
        receipt['status'] = 'binding_failed'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'receipt.json'
            p.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'binding failed'):
                fb.load_bound_cohort(p)

    def test_groups_disjoint_deterministic_and_label_independent(self):
        _, rows, _ = fb.load_bound_cohort(RECEIPT)
        a = fb.instrument_groups(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            row['label']['outcome'] = not row['label']['outcome']
        self.assertEqual(a, fb.instrument_groups(list(reversed(changed))))
        self.assertEqual([len(x) for x in a.values()], [2, 2, 1])
        flattened = [x for group in a.values() for x in group]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_stress_boundary_and_crossing_label(self):
        windows = [{'start_ns': 100, 'end_ns': 200}]
        for start, end, expected in [(80,99,False),(80,100,True),(80,110,True),
                                     (110,120,True),(199,210,True),(200,210,False)]:
            self.assertEqual(fb.overlaps_stress({'decision_ns':start,'label':{'end_ns':end}}, windows), expected)

    def test_calendar_end_date_is_inclusive_proposal(self):
        p = {'regime_holdout_plan': {'candidate_calendar_stress_windows':[
            {'label':'one day','window':'2025-04-30 .. 2025-04-30'}]}}
        w = fb.calendar_windows(p)[0]
        self.assertEqual(w['end_ns'] - w['start_ns'], 86400*10**9)

    def test_no_heldout_instruments_or_stress_rows_fit(self):
        p, rows, folds = fb.load_bound_cohort(RECEIPT)
        groups, windows = fb.instrument_groups(rows), fb.calendar_windows(p)
        planned = fb.plan_folds(rows, folds, groups, windows)
        lookup = {r['id']:r for r in rows}
        for fold in planned:
            for ids in fold['fit_ids'].values():
                for rid in ids:
                    self.assertIn(fb.instrument_key(lookup[rid]), groups['primary'])
                    self.assertFalse(fb.overlaps_stress(lookup[rid], windows))

    def test_feature_matrix_excludes_metadata(self):
        _, rows, _ = fb.load_bound_cohort(RECEIPT)
        order = sorted(rows[0]["features"])
        matrix, labels, ids = fb.feature_matrix(rows[:3], order)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix[0]), 11)
        self.assertEqual(len(labels), len(ids), 3)
        self.assertNotIn("asset_id", order)
        with self.assertRaisesRegex(ValueError, "feature_order"):
            fb.feature_matrix(rows[:1], order + ["asset_id"])

    def test_first_fold_dev_empty_and_regimes_missing_are_blockers(self):
        p = fb.preflight(RECEIPT)
        self.assertEqual(p['status'], 'blocked_before_fit')
        self.assertEqual(p['folds'][0]['fit_counts']['dev'], 0)
        self.assertIn('fold_0_empty_dev_after_holdout', p['block_reasons'])
        self.assertIn('point_in_time_regime_holdout_masks_missing', p['block_reasons'])
        self.assertFalse(p['training_performed'])
        self.assertEqual(p['network_model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
