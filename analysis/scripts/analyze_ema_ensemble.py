"""EMA-weights 5-fold ensemble: the submit gate and the submission file.

  python analysis/scripts/analyze_ema_ensemble.py gate        hold-out, paired vs the released ensemble
  python analysis/scripts/analyze_ema_ensemble.py submission  merge the test chunks into submission.csv

Gate fixed before the run: EMA ensemble hold-out PQ >= released ensemble hold-out PQ (paired over the
50 images; released per-image counts from analysis/results/merge/merge_chunk*.json, full-precision
production pipeline, TP/FP/FN 400/239/211).
"""
import csv
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
R = 'analysis/results/ema_ensemble'


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


def gate():
    ema = {}
    for p in glob.glob(f'{R}/ema_*_chunk*.json'):
        d = json.load(open(p))
        assert d['status'] == 'done', p
        ema.update({r['name']: r['counts'] for r in d['holdout']})
    rel = {}
    for p in glob.glob('analysis/results/merge/merge_chunk*.json'):
        rel.update({r['name']: r['base'] for r in json.load(open(p))['results']})
    names = sorted(rel)
    assert sorted(ema) == names and len(names) == 50, (len(ema), len(names))
    a, b = np.array([ema[n] for n in names], float), np.array([rel[n] for n in names], float)
    assert b.sum(0)[1:].astype(int).tolist() == [400, 239, 211]
    bs = [PQ(a[k].sum(0)) - PQ(b[k].sum(0)) for k in (rng.integers(0, 50, 50) for _ in range(4000))]
    d = PQ(a.sum(0)) - PQ(b.sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = bool(d >= 0)
    print(f'hold-out: released {PQ(b.sum(0)):.4f} EMA ensemble {PQ(a.sum(0)):.4f} diff {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  '
          f'TP/FP/FN released {b.sum(0)[1:].astype(int).tolist()} EMA {a.sum(0)[1:].astype(int).tolist()}')
    print('GATE:', 'PASS -> run the test stage and submit' if ok else 'FAIL -> do not submit')
    json.dump({'released': round(PQ(b.sum(0)), 4), 'ema': round(PQ(a.sum(0)), 4), 'diff': round(d, 4),
               'ci95': [round(lo, 4), round(hi, 4)], 'pass': ok}, open(f'{R}/ema_gate_summary.json', 'w'), indent=1)


def submission():
    rows, stems = [], set()
    for p in sorted(glob.glob(f'{R}/ema_*_chunk*.json')):
        d = json.load(open(p))
        assert d['status'] == 'done', p
        rows += d['rows']
        stems |= set(d['test_stems'])
    sys.path.insert(0, '.')
    import filament_pipeline as fp
    check = fp.validate_submission_rows(rows, stems)
    print(f'{len(stems)} test images, {len(rows)} rows; check: {check}')
    assert len(stems) == 180
    with open(f'{R}/submission_ema.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filament_id', 'segmentation_rle'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {R}/submission_ema.csv')


if __name__ == '__main__':
    {'gate': gate, 'submission': submission}[sys.argv[1]]()
