#!/usr/bin/env python3
"""Check the stored sunspot snapshot and prediction metrics without retraining."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = ROOT / 'SILSO_monthly_1990_2025_snapshot.csv'
meta = json.loads((ROOT / 'metrics_recent.json').read_text(encoding='utf-8'))
assert hashlib.sha256(raw.read_bytes()).hexdigest() == meta['snapshot_sha256'], 'Snapshot SHA256 mismatch'
with raw.open(newline='', encoding='utf-8') as f:
    observations = list(csv.DictReader(f))
assert len(observations) == 432 and observations[0]['date'] == '1990-01' and observations[-1]['date'] == '2025-12'

with (ROOT / 'predictions_recent_all.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
test = [r for r in rows if r['split'] == 'test']
assert len(test) == 60 and test[0]['date'] == '2021-01' and test[-1]['date'] == '2025-12'
for name in ('持续性基线', '前3月均值', '11年滞后', '岭回归AR(36)', 'MLP', 'LSTM', 'GRU'):
    rmse = math.sqrt(sum((float(r['observed']) - float(r[name])) ** 2 for r in test) / len(test))
    original = float(meta['scores'][name]['test']['RMSE'])
    if abs(rmse - original) > 0.001:
        raise AssertionError(f'{name}: stored predictions RMSE {rmse:.6f} != metrics {original:.6f}')
    print(f'{name:14s} RMSE = {rmse:.3f}')
print('OK: source snapshot SHA256, test split, and stored predictions match.')
