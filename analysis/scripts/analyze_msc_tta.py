"""Multi-scale TTA out of fold (analysis/scripts/filament_msc_tta_gpu.py).

Six scale combinations of the released fold models, each scored on the fold's own validation images with
the production candidate rule, refiner and near-limb filter; every combination is compared PAIRED with the
single-scale baseline recomputed in the same run, over the same images.

Rules fixed BEFORE the run:
  primary   the three-scale combination s075_100_125, near-limb counts (the released pipeline's output),
            all 657 images: adopt only if the gain over s100 is >= +0.005 with a paired bootstrap CI95 above 0;
  secondary the other four combinations are reported for context; they decide nothing (five comparisons);
  next      if adopted: (1) 5-model x 3-scale ensemble on the 50 hold-out images must not fall below the
            single-scale ensemble (0.4384 with the limb filter), (2) the instance scorer is refitted on
            multi-scale features (calibration shifts) and must keep its out-of-fold gain, (3) the test set is
            re-run, (4) one leaderboard probe.
  integrity single-scale counts without the limb filter are compared with the stored CPU out-of-fold run.
"""
import glob
import json
import sys

import numpy as np

from scorer_common import PQ, paired_ci

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
PRIMARY = 's075_100_125'

res = []
for p in sorted(glob.glob('analysis/results/msc_tta/msc_fold*.json')):
    d = json.load(open(p))
    assert d['status'] == 'done', (p, d['status'])
    res += d['results']
res.sort(key=lambda r: r['name'])
assert len(res) == 657, len(res)
combos = list(res[0]['combos'])

stored = {}
for p in sorted(glob.glob('analysis/results/oof/oof_fold*.json')):
    stored.update({r['name']: r['with_refiner'] for r in json.load(open(p))['results']})
S0 = np.array([r['combos']['s100']['nolimb'] for r in res], float)
ST = np.array([stored[r['name']] for r in res], float)
same = int((np.abs(S0[:, 1:] - ST[:, 1:]) < 0.5).all(1).sum())
print(f'integrity: single-scale counts equal the stored CPU run on {same}/657 images; pooled PQ {PQ(S0.sum(0)):.4f} vs stored {PQ(ST.sum(0)):.4f}')

out = {'integrity_identical': same, 'combos': {}}
base = {k: np.array([r['combos']['s100'][k] for r in res], float) for k in ('limb', 'nolimb')}
for c in combos:
    row = {}
    line = f'{c:13s}'
    for k in ('limb', 'nolimb'):
        a = np.array([r['combos'][c][k] for r in res], float)
        d, (lo, hi) = paired_ci(a, base[k], rng)
        row[k] = {'pq': round(PQ(a.sum(0)), 4), 'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)],
                  'tp_fp_fn': a.sum(0)[1:].astype(int).tolist()}
        line += f' | {k}: {PQ(a.sum(0)):.4f} diff {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}] TP/FP/FN {row[k]["tp_fp_fn"]}'
    row['mean_candidates'] = round(float(np.mean([r['combos'][c]['n_cand'] for r in res])), 2)
    out['combos'][c] = row
    print(line + f' | candidates/image {row["mean_candidates"]}')

p = out['combos'][PRIMARY]['limb']
ok = bool(p['diff'] >= 0.005 and p['ci95'][0] > 0)
out['verdict'] = ('ADOPT -> hold-out ensemble check, scorer refit on multi-scale features, test re-run, leaderboard probe' if ok else
                  'NOT ADOPTED (gain < +0.005 or CI includes 0)')
print(f'\nprimary {PRIMARY} (limb): {p["diff"]:+.4f}, CI95 [{p["ci95"][0]:+.4f}, {p["ci95"][1]:+.4f}]')
print('VERDICT:', out['verdict'])
json.dump(out, open('analysis/results/msc_tta_summary.json', 'w'), indent=1)
