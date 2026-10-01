"""Label-convention probes (analysis/scripts/filament_label_probe_cpu.py): effect of each probe on the
training annotations, i.e. the NULL prediction for a leaderboard probe (test labelled like training).

For every probe: pooled PQ, SQ (mean IoU of TPs) and RQ on the out-of-fold (657) and hold-out (50)
predictions, the paired difference to the released pipeline with a bootstrap CI95 over images, and the
leaderboard reading expected under the null: 0.363 + 0.83 * diff (0.363 = our hidden score estimated
from the rank inside the 0.36 band, 0.83 = leaderboard / hold-out slope of the earlier submissions).
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
LB0, SLOPE = 0.363, 0.83


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


d = json.load(open('analysis/results/label_probe/label_probe.json'))
assert d['status'] == 'done'
out = {}
for tag in ('oof', 'holdout'):
    res = [r for r in d['results'] if r['set'] == tag]
    n = len(res)
    areas = np.concatenate([r['areas'] for r in res])
    base = np.array([r['rules']['base'] for r in res], float)
    boots = [rng.integers(0, n, n) for _ in range(4000)]
    print(f'== {tag} ({n} images, {len(areas)} instances, median area {np.median(areas):.0f} px): '
          f'base PQ {PQ(base.sum(0)):.4f}  SQ {base[:, 0].sum() / base[:, 1].sum():.4f}  TP/FP/FN {base.sum(0)[1:].astype(int).tolist()}')
    out[tag] = {'base': round(PQ(base.sum(0)), 4), 'rules': {}}
    for v in res[0]['rules']:
        if v == 'base':
            continue
        a = np.array([r['rules'][v] for r in res], float)
        diff = PQ(a.sum(0)) - PQ(base.sum(0))
        lo, hi = np.percentile([PQ(a[k].sum(0)) - PQ(base[k].sum(0)) for k in boots], [2.5, 97.5])
        t = a.sum(0)
        extra = f'  removes {(areas < int(v.split("@")[1])).mean():.1%} of instances' if v.startswith('small@') else ''
        print(f'   {v:11s} PQ {PQ(t):.4f} diff {diff:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  SQ {t[0] / max(t[1], 1):.4f} '
              f'TP/FP/FN {t[1:].astype(int).tolist()}  null LB ~ {LB0 + SLOPE * diff:.3f}{extra}')
        out[tag]['rules'][v] = {'pq': round(PQ(t), 4), 'diff': round(diff, 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                'null_lb': round(LB0 + SLOPE * diff, 3)}
json.dump(out, open('analysis/results/label_probe_summary.json', 'w'), indent=1)
