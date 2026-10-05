"""Independently audit captured prices, frozen means and alternate-line pricing."""
from pathlib import Path
import hashlib
import json
import math
from scipy.stats import nbinom, poisson

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'brazil_corners_betplay_2026_10_05'
BASE = json.loads((PARENT / 'predictions.json').read_text(encoding='utf-8'))
REPORT = json.loads((HERE / 'predictions.json').read_text(encoding='utf-8'))
ROWS = []
for path in sorted(HERE.glob('quotes_*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['period'] == 'full_time' and payload['scope'] == 'total'
    ROWS.extend(payload['rows'])
ROWS.sort(key=lambda row: row[0])
canonical = json.dumps(ROWS, ensure_ascii=False, separators=(',', ':'))
checksum = 2166136261
for char in canonical:
    checksum = ((checksum ^ ord(char)) * 16777619) & 0xffffffff
# These values were computed from the retained read-only browser DOM captures.
assert len(ROWS) == 40 and len({row[0] for row in ROWS}) == 40
assert len(canonical) == 8241 and checksum == 1284861246
assert sum(len(row[4]) * 2 for row in ROWS) == 678
quotes = {}
for event, observed, home, away, lines in ROWS:
    for line, over, under in lines:
        assert line % 1 == .5
        quotes[(str(event), line, 'over')] = over
        quotes[(str(event), line, 'under')] = under
fixtures = {row['fixture_id']: row for row in BASE['fixtures']}
max_diff = dict(probability=0.0, fair_odds=0.0, ev=0.0, expected_profit_cop=0.0)
audited = []
assert REPORT['prediction_as_of'] == BASE['prediction_as_of']
assert REPORT['model_training_cutoff'] == BASE['model_training_cutoff']
assert REPORT['model'] == BASE['model']
assert REPORT['coverage']['complete'] and len(REPORT['markets']) == 678
assert not REPORT['exclusions'] and REPORT['stake_cop'] == 25000
seen = set()
for row in REPORT['markets']:
    key = (str(row['book_event_id']), row['line'], row['side'])
    assert key not in seen and row['odds'] == quotes[key]
    seen.add(key)
    frozen = fixtures[row['fixture_id']]
    assert row['mean'] == frozen['means']['total']
    assert row['alpha'] == frozen['distribution']['total']['alpha']
    assert row['source_url'] == frozen['source_url']
    n = 1 / row['alpha'] if row['alpha'] > 1e-8 else None
    distribution = nbinom(n, n / (n + row['mean'])) if n else poisson(row['mean'])
    k = math.floor(row['line'])
    probability = float(distribution.sf(k) if row['side'] == 'over' else distribution.cdf(k))
    expected = dict(probability=probability, fair_odds=1 / probability,
                    ev=probability * row['odds'] - 1,
                    expected_profit_cop=25000 * (probability * row['odds'] - 1))
    actual = dict(probability=row['positive_profit_probability'], fair_odds=row['fair_odds'],
                  ev=row['ev'], expected_profit_cop=row['expected_profit_cop'])
    for field, value in expected.items():
        max_diff[field] = max(max_diff[field], abs(value - actual[field]))
        assert math.isclose(value, actual[field], rel_tol=1e-10, abs_tol=1e-7)
    assert row['probabilities']['push'] == 0
    assert row['positive_ev'] == (expected['ev'] > 0)
    assert row['positive_profit_probability_at_least_60pct'] == (probability >= .6)
    assert row['gross_payout_cop_by_settlement']['full_win'] == 25000 * row['odds']
    assert row['net_profit_cop_by_settlement']['full_loss'] == -25000
    audited.append(dict(fixture_id=row['fixture_id'], book_event_id=row['book_event_id'],
                        side=row['side'], line=row['line'], odds=row['odds'], **expected))
assert seen == set(quotes)
paired = {(row['book_event_id'], row['line']): {} for row in REPORT['markets']}
for row in REPORT['markets']:
    paired[(row['book_event_id'], row['line'])][row['side']] = row['positive_profit_probability']
assert len(paired) == 339
assert all(math.isclose(pair['over'] + pair['under'], 1, abs_tol=1e-12) for pair in paired.values())
positives = sorted([row for row in REPORT['markets'] if row['ev'] > 0],
                   key=lambda row: (-row['ev'], row['fixture_id'], row['side'], row['line']))
assert positives == REPORT['positive_ev_markets']
for field, allowed in [('best_positive_ev_per_fixture', positives),
                       ('best_positive_ev_at_least_60pct_per_fixture',
                        [row for row in positives if row['positive_profit_probability'] >= .6])]:
    best = {}
    for row in allowed:
        best.setdefault(row['fixture_id'], row)
    assert list(best.values()) == REPORT[field]
result = dict(passed=True, fixtures=40, selections=678, complementary_pairs=339,
              browser_to_saved_price_verification=dict(canonical_characters=len(canonical),
                   fnv1a32=checksum, exact_capture_checksum_match=True,
                   note='FNV is a transcription check; SHA256 below preserves saved evidence.'),
              positive_selections=len(positives), positive_fixtures=len(REPORT['best_positive_ev_per_fixture']),
              positive_fixtures_at_least_60pct=len(REPORT['best_positive_ev_at_least_60pct_per_fixture']),
              maximum_absolute_differences=max_diff,
              sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [HERE/'predictions.json', PARENT/'predictions.json',
                                   Path(REPORT['model']), *sorted(HERE.glob('quotes_*.json'))]},
              independent_calculations=audited)
(HERE/'independent_math_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items()
                  if key not in ('independent_calculations', 'sha256')}, indent=2))
