#!/usr/bin/env python3
"""Parse SILSO SN_m_tot_V2.0.txt, or download current data, to a candidate CSV.

Produces a NEW file. It never overwrites the fixed study snapshot.
Because SILSO makes revisions, current official values can differ from the
frozen 2026-10-24 snapshot and should not be mixed into published metrics.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import urllib.request
import pandas as pd

OFFICIAL = 'https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt'
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument('--source', type=Path, help='Downloaded SILSO monthly raw TXT')
g.add_argument('--download', action='store_true', help='Download current official raw TXT')
p.add_argument('--output', type=Path, default=HERE/'candidate_1990_2025.csv')
p.add_argument('--compare', type=Path, default=HERE/'SILSO_monthly_1990_2025_snapshot.csv')
args = p.parse_args()
raw = urllib.request.urlopen(OFFICIAL, timeout=30).read() if args.download else args.source.read_bytes()
rows = []
for lineno, line in enumerate(raw.decode('utf8').splitlines(), 1):
    fields = line.split()
    if not fields:
        continue
    if len(fields) < 4:
        raise ValueError(f'Bad SILSO row {lineno}: {line!r}')
    year, month = int(fields[0]), int(fields[1])
    if year < 1990 or year > 2025:
        continue
    if not 1 <= month <= 12:
        raise ValueError(f'Bad month row {lineno}')
    number = float(fields[3])
    if number < 0:
        raise ValueError(f'Missing/unavailable observation at {year}-{month:02d}')
    rows.append({'date':f'{year:04d}-{month:02d}', 'year':year, 'month':month, 'sunspot_number':number})
df = pd.DataFrame(rows).sort_values('date')
assert len(df) == 432, f'Expected 432 months; got {len(df)}'
assert df.date.tolist() == pd.period_range('1990-01', '2025-12', freq='M').astype(str).tolist()
args.output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output, index=False, float_format='%.1f')
print('Wrote candidate:', args.output, 'rows:', len(df))
print('Official raw SHA256:', hashlib.sha256(raw).hexdigest())
print('Generated UTC:', datetime.datetime.now(datetime.timezone.utc).isoformat())
if args.compare.exists():
    original = pd.read_csv(args.compare)
    j = df.merge(original[['date','sunspot_number']], on='date', suffixes=('_current','_fixed'))
    bad = j[(j.sunspot_number_current - j.sunspot_number_fixed).abs() > 1.e-6]
    print('Months different from fixed 2026-10-24 study snapshot:', len(bad))
    if len(bad):
        print(bad.head(10).to_string(index=False))
