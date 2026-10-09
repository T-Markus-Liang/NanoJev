import json
from pathlib import Path
import tempfile
import unittest

import financial_baselines_v1 as baselines
import financial_risk_v1 as risk
import financial_real_data_path_v1 as rp
import financial_real_data_risk_v1 as gated
import run_financial_risk_gate_v1 as runner
import validate_financial_risk_gate_protocol_v1 as validator

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'results/financial_pit_r1_bound_receipt_20260920_v5.json'
PROTOCOL = ROOT / 'research/financial_risk_gate_protocol_v1.json'
FROZEN = ROOT / 'research/financial_r1_frozen_parameters_v1.json'
AUTH = ROOT / 'results/financial_risk_gate_authorization_20260921_v1.json'
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


def make_decision_record(ns, verdict, codes=(), requested=1.0, allowed=None, outcome=None):
    return {'decision_id': f'd::{ns}', 'decision_ns': ns, 'risk_decision': verdict,
            'risk_reason_codes': list(codes), 'requested_quantity': requested,
            'risk_allowed_quantity': allowed, 'outcome': outcome or verdict}


class FrozenLimitTests(unittest.TestCase):
    def test_frozen_limits_equal_engine_defaults(self):
        limits = gated.load_frozen_risk_limits()
        self.assertEqual(limits.to_dict(), risk.RiskLimits().to_dict())
        self.assertEqual(limits.policy_version, risk.RISK_POLICY_VERSION)

    def test_missing_frozen_limit_fails_closed(self):
        document = json.loads(FROZEN.read_text())
        del document['simulator_and_risk_parameters']['limits.max_net_loss_notional']
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'frozen.json'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(gated.RiskGatePathError, 'missing'):
                gated.load_frozen_risk_limits(path)

    def test_extra_frozen_limit_fails_closed(self):
        document = json.loads(FROZEN.read_text())
        document['simulator_and_risk_parameters']['limits.invented_limit'] = 1.0
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'frozen.json'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(gated.RiskGatePathError, 'extra'):
                gated.load_frozen_risk_limits(path)

    def test_factory_produces_independent_engines(self):
        factory = gated.frozen_gate_factory()
        first, second = factory(), factory()
        self.assertIsNot(first, second)
        self.assertIsNot(first.kill_switch, second.kill_switch)
        self.assertEqual(first.limits.to_dict(), second.limits.to_dict())


class GateStatsTests(unittest.TestCase):
    def test_stats_count_verdicts_and_reductions(self):
        receipt = {'decisions': [
            make_decision_record(1, 'allow', ('ok',), allowed=1.0),
            make_decision_record(2, 'reduce', ('order_notional_limit',), allowed=0.4),
            make_decision_record(3, 'block', ('position_limit',), allowed=0.0,
                                 outcome='risk_blocked'),
            make_decision_record(4, None)]}
        stats = gated.gate_decision_stats(receipt)
        self.assertEqual(stats['evaluated_decisions'], 3)
        self.assertEqual((stats['allowed_decisions'], stats['reduced_decisions'],
                          stats['blocked_decisions']), (1, 1, 1))
        self.assertAlmostEqual(stats['reduced_quantity_total'], 0.6)
        self.assertEqual(stats['reason_code_counts'],
                         {'ok': 1, 'order_notional_limit': 1, 'position_limit': 1})
        self.assertEqual(len(stats['non_allow_records']), 2)
        self.assertEqual(stats['non_allow_records'][1]['outcome'], 'risk_blocked')

    def test_undeclared_verdict_fails_closed(self):
        receipt = {'decisions': [make_decision_record(1, 'maybe')]}
        with self.assertRaisesRegex(gated.RiskGatePathError, 'undeclared'):
            gated.gate_decision_stats(receipt)

    def test_regime_attribution_uses_overlapping_strata(self):
        rows = [make_row(f'r{i}', 'X-PERP', 'binance_um', (i + 1) * 2 * DAY)
                for i in range(3)]
        masks = {rows[0]['id']: {'regimes': [rp.REGIME_HIGH_VOL], 'evaluable': []},
                 rows[1]['id']: {'regimes': [], 'evaluable': []},
                 rows[2]['id']: {'regimes': [rp.REGIME_HIGH_VOL,
                                           rp.REGIME_THIN_LIQUIDITY], 'evaluable': []}}
        records = [make_decision_record(rows[0]['decision_ns'], 'block'),
                   make_decision_record(rows[1]['decision_ns'], 'allow'),
                   make_decision_record(rows[2]['decision_ns'], 'reduce'),
                   make_decision_record(5, 'block')]  # no record -> pre-first bucket
        buckets = gated.gate_regime_attribution(records, rows, masks)
        self.assertEqual(buckets[rp.REGIME_HIGH_VOL],
                         {'evaluated_decisions': 2, 'allowed': 0,
                          'reduced': 1, 'blocked': 1})
        self.assertEqual(buckets[rp.REGIME_THIN_LIQUIDITY]['reduced'], 1)
        self.assertEqual(buckets[rp.REGIME_NONE]['allowed'], 1)
        self.assertEqual(buckets[rp.REGIME_PRE_FIRST]['blocked'], 1)


class GatedCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1_protocol, cls.rows, cls.folds = baselines.load_bound_cohort(RECEIPT)
        cls.groups = baselines.instrument_groups(cls.rows)
        cls.masks = rp.compute_regime_masks(cls.rows)
        cls.regimes = rp.stress_regime_windows(cls.r1_protocol)
        cls.by_id = {row['id']: row for row in cls.rows}

    def _cell(self, asset, policy, fold_index=0):
        fold = self.folds[fold_index]
        path = rp.build_cell_path(self.by_id, fold['retained']['test'], asset,
                                  'binance_um', regimes=self.regimes,
                                  asof_ns=fold['windows']['test'][1])
        return gated.run_gated_cell(path, policy)

    def test_no_trade_cell_stays_flat_under_gate(self):
        cell = self._cell('SOLUSDT-PERP', 'no_trade')
        self.assertEqual(cell['counts']['fills'], 0)
        self.assertEqual(cell['counts']['decisions'], 0)
        self.assertEqual(cell['gate_stats']['evaluated_decisions'], 0)
        self.assertTrue(cell['byte_identical'])
        self.assertEqual({p['equity'] for p in cell['equity_curve']}, {rp.INITIAL_CASH})
        self.assertEqual(cell['risk']['risk_gate'], 'financial_risk_v1')
        self.assertEqual(cell['risk']['limits'], risk.RiskLimits().to_dict())

    def test_frozen_gate_blocks_every_short_intent(self):
        cell = self._cell('SOLUSDT-PERP', 'unit_short_always')
        stats = cell['gate_stats']
        self.assertGreater(stats['evaluated_decisions'], 0)
        self.assertEqual(stats['blocked_decisions'], stats['evaluated_decisions'])
        self.assertEqual(cell['counts']['risk_blocked_decisions'],
                         stats['evaluated_decisions'])
        self.assertEqual(cell['counts']['fills'], 0)
        self.assertEqual(set(stats['reason_code_counts']), {'position_limit'})
        self.assertTrue(cell['byte_identical'])
        for record in stats['non_allow_records']:
            self.assertEqual(record['risk_allowed_quantity'], 0.0)
            self.assertEqual(record['outcome'], 'risk_blocked')

    def test_btc_long_is_reduced_then_blocked_at_exposure_caps(self):
        cell = self._cell('BTCUSDT-PERP', 'unit_long_always')
        stats = cell['gate_stats']
        self.assertGreater(stats['reduced_decisions'], 0)
        self.assertGreater(stats['blocked_decisions'], 0)
        self.assertIn('order_notional_limit', stats['reason_code_counts'])
        self.assertTrue({'gross_exposure_limit', 'net_exposure_limit'}
                        <= set(stats['reason_code_counts']))
        self.assertGreater(stats['reduced_quantity_total'], 0.0)
        self.assertGreater(cell['counts']['fills'], 0)
        self.assertTrue(cell['byte_identical'])
        # A reduce verdict still executes at the allowed quantity.
        reduced = [r for r in stats['non_allow_records']
                   if r['risk_decision'] == 'reduce']
        self.assertTrue(all(0.0 < r['risk_allowed_quantity'] < r['requested_quantity']
                            for r in reduced))

    def test_cell_carries_t12_field_set_plus_gate_fields(self):
        cell = self._cell('ETHUSDT-PERP', 'unit_long_always')
        for field in ('policy', 'decision_count', 'backtest_sha256', 'byte_identical',
                      'counts', 'risk', 'equity_curve', 'equity_curve_fingerprint',
                      'drawdown', 'margin_usage', 'turnover', 'costs', 'funding',
                      'per_regime_calendar_stress', 'attribution', 'market_path',
                      'gate_stats', 'determinism'):
            self.assertIn(field, cell)
        self.assertTrue(cell['determinism']['fresh_gate_per_replay'])
        self.assertEqual(len(cell['determinism']['backtest_sha256']), 2)
        self.assertEqual(cell['determinism']['backtest_sha256'][0],
                         cell['backtest_sha256'])
        self.assertLess(abs(cell['attribution']['instrument_residual']), 1e-4)

    def test_gate_summary_aggregates_counts_not_pnl(self):
        fold = gated.run_gated_fold(self.folds[0], 0, self.rows, self.groups,
                                    self.masks, self.regimes, policies=('no_trade',))
        summary = gated.summarize_gate([fold])
        for policy_stats in (v['policies'] for v in
                             summary['per_instrument'].values()):
            self.assertEqual(set(policy_stats), {'no_trade'})
        self.assertEqual(summary['totals']['evaluated_decisions'], 0)


class ValidatorTests(unittest.TestCase):
    def test_preflight_passes_on_real_artifacts(self):
        result = validator.validate(PROTOCOL, RECEIPT)
        self.assertEqual(result['status'], 'protocol_valid_not_replay_authorized')
        self.assertEqual(result['cell_count'], 15)
        self.assertEqual(result['gate']['lifecycle'], 'fresh_risk_engine_per_replay')
        self.assertFalse(result['authorization']['replay_authorized'])

    def test_tampered_protocol_rejected(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol['risk_gate']['limits']['max_net_loss_notional'] = 1.0
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'protocol.json'
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, 'pinned review artifact'):
                validator.validate(path, RECEIPT)

    def test_declared_limits_must_match_frozen(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol['risk_gate']['limits']['max_net_loss_notional'] = 1.0
        with self.assertRaisesRegex(ValueError, 'frozen R1 limits'):
            validator._check_gate_declaration(protocol)

    def test_wrong_t12_binding_rejected(self):
        protocol = json.loads(PROTOCOL.read_text())
        amendment = dict(protocol['amendment_of'])
        amendment['protocol_path'] = 'research/financial_experiment_protocol_v1.json'
        with self.assertRaisesRegex(ValueError, 'T12 protocol'):
            validator._check_t12_binding(amendment)
        amendment = dict(protocol['amendment_of'])
        amendment['protocol_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validator._check_t12_binding(amendment)


class RunnerTests(unittest.TestCase):
    def test_blocked_without_authorization(self):
        result = runner.run(PROTOCOL, RECEIPT, None)
        self.assertEqual(result['status'], 'blocked_replay_authorization_missing')
        self.assertFalse(result['replay_performed'])
        self.assertFalse(result['real_replay_performed'])
        self.assertFalse(result['headline_allowed'])
        self.assertEqual(len(result['cell_plan']), 15)

    def test_wrong_schema_authorization_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            auth = Path(tmp) / 'auth.json'
            auth.write_text(json.dumps({'schema_version': 'bogus'}))
            with self.assertRaisesRegex(ValueError, 'schema'):
                runner.run(PROTOCOL, RECEIPT, auth)

    def test_authorization_must_bind_t13_protocol(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            auth = Path(tmp) / 'auth.json'
            auth.write_text(json.dumps({
                'schema_version': runner.AUTH_SCHEMA,
                'protocol_sha256': '0' * 64,
                'decision': 'approved_for_real_replay'}))
            with self.assertRaisesRegex(ValueError, 'different T13 protocol'):
                runner.run(PROTOCOL, RECEIPT, auth)

    def test_t12_authorization_does_not_authorize_t13(self):
        t12_auth = ROOT / 'results/financial_validation_authorization_20260921_v1.json'
        with self.assertRaisesRegex(ValueError, 'different T13 protocol'):
            runner.run(PROTOCOL, RECEIPT, t12_auth)


if __name__ == '__main__':
    unittest.main()
