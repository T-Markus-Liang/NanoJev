import json
from pathlib import Path
import tempfile
import unittest

import financial_backtest_v1 as backtest
import financial_baselines_v1 as baselines
import financial_real_data_path_v1 as rp
import run_financial_validation_v1 as runner
import validate_financial_validation_protocol_v1 as validator

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'results/financial_pit_r1_bound_receipt_20260920_v5.json'
PROTOCOL = ROOT / 'research/financial_validation_protocol_v1.json'
DAY = rp.DAY_NS
DEF_HASH = 'cffd49217c95e83bedffc05f43f018964758cc58ba153f3963fc221d1f37124c'
FEATURES = {
    'funding_interval_hours': 8, 'index_price': 100.0, 'last_funding_rate': 0.0001,
    'mark_index_basis_bps': 1.0, 'mark_price': 100.0, 'open_interest_level': 1e6,
    'open_interest_log_change_1d': 0.0, 'quote_volume': 1e6,
    'realized_vol_24bar': 0.01, 'taker_buy_ratio': 0.5, 'trade_count': 1000,
}


def make_row(rid, asset, venue, decision_ns, outcome=False, **overrides):
    values = dict(FEATURES)
    values.update(overrides)
    return {
        'schema_version': 'nanojev-financial-pit-v1',
        'id': rid, 'asset_id': asset, 'venue': venue,
        'decision_ns': decision_ns, 'universe_available_ns': decision_ns,
        'features': {name: {'value': value, 'event_ns': decision_ns - 1,
                            'available_ns': decision_ns, 'fit_cutoff_ns': 0,
                            'source_id': 'test', 'version': 'v1'}
                     for name, value in values.items()},
        'label': {'event': 'perp_forward_mark_return_up_25bps_1d_gross',
                  'definition_sha256': DEF_HASH,
                  'end_ns': decision_ns + DAY - 1,
                  'available_ns': decision_ns + 8 * DAY - 1,
                  'outcome': outcome}}


class QuoteConstructionTests(unittest.TestCase):
    def test_quotes_collapse_book_to_mark_and_carry_funding(self):
        row = make_row('a', 'X-PERP', 'binance_um', 10**15,
                       mark_price=200.0, quote_volume=1e9, last_funding_rate=0.0003)
        (quote,) = rp.records_to_quotes([row])
        self.assertEqual(quote.bid, quote.ask, 200.0)
        self.assertEqual(quote.volume, 1e9 / 200.0)
        self.assertEqual(quote.funding_rate, 0.0003)
        self.assertEqual(quote.mark_price, 200.0)
        self.assertIsNone(quote.last_price)
        self.assertEqual(quote.available_ns, 10**15)
        self.assertEqual(quote.event_ns, 10**15 - 1)
        self.assertEqual(quote.source_id, 'r1_frozen_cohort_record')

    def test_quotes_reject_nonpositive_mark(self):
        row = make_row('a', 'X-PERP', 'binance_um', 10**15, mark_price=-1.0)
        with self.assertRaisesRegex(rp.RealDataPathError, 'non-positive'):
            rp.records_to_quotes([row])


class RegimeMaskTests(unittest.TestCase):
    def _rows(self, n, value=0.01, feature='realized_vol_24bar', asset='X-PERP'):
        return [make_row(f'r{i}', asset, 'binance_um', (i + 1) * 2 * DAY,
                         **{feature: value}) for i in range(n)]

    def test_warmup_masks_do_not_vote_before_min_history(self):
        masks = rp.compute_regime_masks(self._rows(rp.MASK_MIN_HISTORY))
        self.assertTrue(all(not entry['evaluable'] for entry in masks.values()))
        self.assertTrue(all(not entry['regimes'] for entry in masks.values()))

    def test_top_decile_flags_extreme_after_warmup(self):
        rows = self._rows(40, value=0.01)
        spike = make_row('spike', 'X-PERP', 'binance_um', 82 * DAY,
                         realized_vol_24bar=9.9)
        masks = rp.compute_regime_masks(rows + [spike])
        self.assertIn(rp.REGIME_HIGH_VOL, masks['spike']['regimes'])
        self.assertEqual(masks['r0']['regimes'], [])

    def test_strictly_prior_history_and_same_timestamp_isolation(self):
        # A spike at an EARLY timestamp must not leak into later rows' history.
        early = make_row('early', 'X-PERP', 'binance_um', 2 * DAY,
                         realized_vol_24bar=9.9)
        rows = self._rows(40, value=0.01)
        masks = rp.compute_regime_masks([early] + rows)
        self.assertEqual(masks['early']['regimes'], [])
        # Same-timestamp records never see each other: two spikes sharing a
        # decision_ns after warmup both evaluate against the pre-spike history.
        pair = [make_row('sa', 'X-PERP', 'binance_um', 100 * DAY,
                         realized_vol_24bar=9.9),
                make_row('sb', 'Y-PERP', 'binance_um', 100 * DAY,
                         realized_vol_24bar=9.9)]
        masks = rp.compute_regime_masks(rows + pair)
        self.assertIn(rp.REGIME_HIGH_VOL, masks['sa']['regimes'])
        self.assertIn(rp.REGIME_HIGH_VOL, masks['sb']['regimes'])

    def test_tie_rule_is_strict(self):
        # Identical values: the 31st row's value equals its own quantile and
        # must NOT enter the regime.
        rows = self._rows(31, value=0.02, feature='mark_index_basis_bps')
        masks = rp.compute_regime_masks(rows)
        self.assertNotIn(rp.REGIME_BASIS_BLOWOUT, masks['r30']['regimes'])
        self.assertIn(rp.REGIME_BASIS_BLOWOUT, masks['r30']['evaluable'])

    def test_thin_liquidity_needs_own_instrument_history(self):
        rows = [make_row(f'r{i}', 'X-PERP', 'binance_um', (i + 1) * 2 * DAY,
                         quote_volume=1e6) for i in range(40)]
        masks = rp.compute_regime_masks(rows)
        # 30 prior own rows + 30 pooled medians: the mask cannot vote at r30.
        self.assertNotIn(rp.REGIME_THIN_LIQUIDITY, masks['r30']['evaluable'])
        # A sustained low-volume stretch pulls the trailing median below the
        # expanding quintile of a high-volume history.
        rows = [make_row(f'r{i}', 'X-PERP', 'binance_um', (i + 1) * 2 * DAY,
                         quote_volume=1e6 if i < 50 else 100.0)
                for i in range(70)]
        masks = rp.compute_regime_masks(rows)
        self.assertIn(rp.REGIME_THIN_LIQUIDITY, masks['r65']['regimes'])
        self.assertNotIn(rp.REGIME_THIN_LIQUIDITY, masks['r40']['regimes'])

    def test_masks_deterministic_under_reordering(self):
        rows = self._rows(80)
        self.assertEqual(rp.compute_regime_masks(rows),
                         rp.compute_regime_masks(list(reversed(rows))))

    def test_mask_stats_counts(self):
        stats = rp.mask_stats(rp.compute_regime_masks(self._rows(80)))
        self.assertEqual(set(stats), set(rp.REGIME_IDS))
        self.assertEqual(stats[rp.REGIME_HIGH_VOL]['evaluable'], 80 - rp.MASK_MIN_HISTORY)


class PolicyAndPathTests(unittest.TestCase):
    def test_reference_decisions_contract(self):
        rows = [make_row(f'r{i}', 'X-PERP', 'binance_um', (i + 1) * 2 * DAY)
                for i in range(5)]
        path = rp.RealCohortPath(quotes=rp.records_to_quotes(rows), regimes=(),
                                 asof_ns=12 * DAY, instruments=(('X-PERP', 'binance_um'),),
                                 ticks=5, tick_ns=DAY, seed=rp.DATA_SEED)
        self.assertEqual(rp.reference_decisions('no_trade', path), [])
        decisions = rp.reference_decisions('unit_long_always', path)
        self.assertEqual(len(decisions), 5)
        self.assertTrue(all(d.action == 'long' and d.quantity == 1.0 for d in decisions))
        self.assertTrue(all(d.decision_ns == q.available_ns
                            for d, q in zip(decisions, path.quotes)))
        shorts = rp.reference_decisions('unit_short_always', path)
        self.assertTrue(all(d.action == 'short' for d in shorts))
        with self.assertRaisesRegex(ValueError, 'fail-closed'):
            rp.reference_decisions('invented_probability_policy', path)

    def test_real_path_summary_is_not_synthetic(self):
        rows = [make_row('r0', 'X-PERP', 'binance_um', 2 * DAY)]
        path = rp.RealCohortPath(quotes=rp.records_to_quotes(rows), regimes=(),
                                 asof_ns=4 * DAY, instruments=(('X-PERP', 'binance_um'),),
                                 ticks=1, tick_ns=DAY, seed=rp.DATA_SEED,
                                 provenance={'dataset_sha256': 'x'})
        summary = path.to_summary()
        self.assertFalse(summary['synthetic'])
        self.assertEqual(summary['data_class'], 'real_frozen_r1_pit_cohort')
        self.assertIsInstance(path, backtest.SyntheticMarketPath)

    def test_execution_policy_matches_frozen_and_declared(self):
        flat = validator._flatten(rp.declared_execution_policy().to_dict())
        frozen = json.loads(
            (ROOT / 'research/financial_r1_frozen_parameters_v1.json').read_text())
        protocol = json.loads(PROTOCOL.read_text())['execution_policy']
        declared = protocol['values']
        for key, expected in declared.items():
            self.assertEqual(flat[key], expected, key)
        deltas = set(protocol['declared_deltas'])
        for key, expected in frozen['simulator_and_risk_parameters'].items():
            if key.startswith('limits.'):
                continue  # risk limits are T13 scope; no gate attached in v1
            if key in deltas:
                continue  # declared real-path deltas, pinned by the protocol itself
            if key in flat:
                if isinstance(expected, list):
                    self.assertEqual(list(flat[key]), expected, key)
                else:
                    self.assertEqual(flat[key], expected, key)

    def test_parse_instrument_key(self):
        self.assertEqual(rp.parse_instrument_key('binance_um:linear:SOLUSDT-PERP'),
                         ('binance_um', 'SOLUSDT-PERP'))
        with self.assertRaises(rp.RealDataPathError):
            rp.parse_instrument_key('binance_um:inverse:BTCUSDT-PERP')


class AttributionTests(unittest.TestCase):
    def test_segment_attribution_to_midpoint_active_record(self):
        rows = [make_row(f'r{i}', 'X-PERP', 'binance_um', (i + 1) * 2 * DAY)
                for i in range(4)]
        masks = {rows[0]['id']: {'regimes': [rp.REGIME_HIGH_VOL], 'evaluable': []},
                 rows[1]['id']: {'regimes': [], 'evaluable': []},
                 rows[2]['id']: {'regimes': [rp.REGIME_HIGH_VOL,
                                           rp.REGIME_BASIS_BLOWOUT], 'evaluable': []},
                 rows[3]['id']: {'regimes': [], 'evaluable': []}}
        points = [{'ns': rows[i]['decision_ns'], 'equity': 100.0 + i,
                   'funding_cost_to_date': 0.0, 'fees_to_date': 0.0,
                   'turnover_to_date': 0.0, 'fill_count': 0, 'liquidation_count': 0}
                  for i in range(4)]
        buckets = rp.pit_regime_attribution(points, rows, masks)
        self.assertEqual(buckets[rp.REGIME_HIGH_VOL]['segments'], 2)
        self.assertEqual(buckets[rp.REGIME_HIGH_VOL]['equity_change'], 2.0)
        self.assertEqual(buckets[rp.REGIME_BASIS_BLOWOUT]['segments'], 1)
        self.assertEqual(buckets[rp.REGIME_NONE]['segments'], 1)


class RealCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.rows, cls.folds = baselines.load_bound_cohort(RECEIPT)
        cls.groups = baselines.instrument_groups(cls.rows)
        cls.masks = rp.compute_regime_masks(cls.rows)
        cls.regimes = rp.stress_regime_windows(cls.protocol)

    def test_masks_cover_cohort(self):
        stats = rp.mask_stats(self.masks)
        for regime_id in rp.REGIME_IDS:
            self.assertGreater(stats[regime_id]['evaluable'], 3000)
            self.assertGreater(stats[regime_id]['member'], 0)

    def test_no_trade_cell_is_flat_control(self):
        by_id = {row['id']: row for row in self.rows}
        fold = self.folds[0]
        path = rp.build_cell_path(by_id, fold['retained']['test'],
                                  'SOLUSDT-PERP', 'binance_um', regimes=self.regimes,
                                  asof_ns=fold['windows']['test'][1])
        cell = rp.run_cell(path, 'no_trade')
        self.assertEqual(cell['counts']['fills'], 0)
        self.assertEqual(cell['counts']['orders'], 0)
        self.assertEqual(cell['counts']['decisions'], 0)
        self.assertTrue(cell['byte_identical'])
        equities = {point['equity'] for point in cell['equity_curve']}
        self.assertEqual(equities, {rp.INITIAL_CASH})
        self.assertEqual(cell['net_pnl_simulated_descriptive'], 0.0)
        self.assertFalse(cell['market_path']['synthetic'])

    def test_unit_long_cell_flows_through_harness(self):
        by_id = {row['id']: row for row in self.rows}
        fold = self.folds[0]
        path = rp.build_cell_path(by_id, fold['retained']['test'],
                                  'SOLUSDT-PERP', 'binance_um', regimes=self.regimes,
                                  asof_ns=fold['windows']['test'][1])
        cell = rp.run_cell(path, 'unit_long_always')
        self.assertGreaterEqual(cell['counts']['fills'], 1)
        self.assertGreater(cell['counts']['funding_payments'], 0)
        self.assertTrue(cell['byte_identical'])
        self.assertEqual(cell['counts']['risk_blocked_decisions'], 0)
        self.assertLess(abs(cell['attribution']['instrument_residual']), 1e-4)
        self.assertLess(abs(cell['attribution']['regime_residual']), 1e-4)
        self.assertEqual(cell['equity_curve'][0]['equity'], rp.INITIAL_CASH)
        self.assertAlmostEqual(cell['equity_curve'][-1]['equity'],
                               cell['final_equity'], places=6)
        self.assertIn('unlabeled', cell['per_regime_calendar_stress'])

    def test_empty_cell_path_rejected(self):
        by_id = {row['id']: row for row in self.rows}
        with self.assertRaisesRegex(rp.RealDataPathError, 'no retained test records'):
            rp.build_cell_path(by_id, self.folds[0]['retained']['test'],
                               'BTCUSDT-PERP', 'bybit_fake', regimes=(), asof_ns=1)

    def test_run_fold_emits_all_declared_cells(self):
        report = rp.run_fold(self.folds[0], 0, self.rows, self.groups, self.masks,
                             self.regimes, policies=('no_trade',))
        self.assertEqual(len(report['cells']), 5)
        groups = {cell['instrument_group'] for cell in report['cells']}
        self.assertEqual(groups, {'primary', 'holdout_instrument_A',
                                  'holdout_instrument_B'})
        for cell in report['cells']:
            self.assertIn('event_outcome_strata', cell)
            self.assertIn('pit_regime_attribution', cell)


class ValidatorTests(unittest.TestCase):
    def test_preflight_passes_on_real_artifacts(self):
        result = validator.validate(PROTOCOL, RECEIPT)
        self.assertEqual(result['status'], 'protocol_valid_not_replay_authorized')
        self.assertEqual(result['cell_count'], 15)
        self.assertEqual(len(result['folds']), 3)
        self.assertFalse(result['authorization']['replay_authorized'])

    def test_tampered_protocol_rejected(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol['cohort']['records'] = 9999
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / 'protocol.json'
            p.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, 'pinned review artifact'):
                validator.validate(p, RECEIPT)

    def test_tampered_t11_receipt_rejected(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol['upstream_t11']['fit_receipt_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / 'protocol.json'
            p.write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                validator.validate(p, RECEIPT)


class RunnerTests(unittest.TestCase):
    def test_blocked_without_authorization(self):
        result = runner.run(PROTOCOL, RECEIPT, None)
        self.assertEqual(result['status'], 'blocked_replay_authorization_missing')
        self.assertFalse(result['replay_performed'])
        self.assertFalse(result['real_replay_performed'])
        self.assertFalse(result['headline_allowed'])
        self.assertEqual(result['cell_plan'] and len(result['cell_plan']), 15)

    def test_wrong_schema_authorization_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            auth = Path(tmp) / 'auth.json'
            auth.write_text(json.dumps({'schema_version': 'bogus'}))
            with self.assertRaisesRegex(ValueError, 'schema'):
                runner.run(PROTOCOL, RECEIPT, auth)

    def test_authorization_must_bind_protocol(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            auth = Path(tmp) / 'auth.json'
            auth.write_text(json.dumps({
                'schema_version': runner.AUTH_SCHEMA,
                'protocol_sha256': '0' * 64,
                'decision': 'approved_for_real_replay'}))
            with self.assertRaisesRegex(ValueError, 'different T12 protocol'):
                runner.run(PROTOCOL, RECEIPT, auth)


if __name__ == '__main__':
    unittest.main()
