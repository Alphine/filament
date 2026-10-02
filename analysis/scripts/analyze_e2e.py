"""End-to-end check of the released code path with the instance scorer (filament_e2e_check_cpu.py).

Compares the kernel's `Pipeline.segment(..., scorer=InstanceScorer.load())` output with what the analysis
predicted for the same scorer:
  hold-out  per-image official counts against those of the stored hold-out predictions after the scorer's
            drop rule (analyze_instance_scorer.py); pooled PQ is compared with 0.4427;
  test      the RLE rows of the test images the kernel processed against the submitted file (argv[1]).
Usage: python analyze_e2e.py [submission_scorer.csv]
"""
import csv
import glob
import json
import sys

import numpy as np

from scorer_common import PQ, counts_after, feature_names, table

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import filament_scorer as fs

chunks = [json.load(open(p)) for p in sorted(glob.glob('analysis/results/e2e/e2e_chunk*.json'))]
assert len(chunks) == 4 and all(c['status'] == 'done' for c in chunks), [c['status'] for c in chunks]
print('environment on the Kaggle image:', chunks[0]['versions'], '| test suite return codes:', [c['tests']['returncode'] for c in chunks])
got = {r['name']: np.array(r['counts'], float) for c in chunks for r in c['holdout']}

oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))
H = table([r for r in oi['results'] if r['set'] == 'holdout'], feature_names(oi['results']))
sc = fs.InstanceScorer.load()
keep = np.array([v >= sc.tau for v in sc.booster.predict(H['X'])])
want = counts_after(H, keep)
names = [r['name'] for r in H['res']]
assert sorted(got) == sorted(names) and len(names) == 50, (len(got), len(names))
G = np.array([got[n] for n in names])
exact = [bool(np.allclose(G[i], want[i], atol=1e-6)) for i in range(len(names))]
print(f'hold-out: per-image counts identical to the analysis on {sum(exact)}/50 images; pooled PQ {PQ(G.sum(0)):.4f} '
      f'(analysis {PQ(want.sum(0)):.4f}, TP/FP/FN {G.sum(0)[1:].astype(int).tolist()} vs {want.sum(0)[1:].astype(int).tolist()})')
for i, ok in enumerate(exact):
    if not ok:
        print('   differs:', names[i], G[i].tolist(), 'expected', want[i].tolist())

ok_test = None
if len(sys.argv) > 1:
    ref = {}
    for r in csv.DictReader(open(sys.argv[1])):
        ref.setdefault(r['filament_id'].rsplit('_', 1)[0], []).append(r['segmentation_rle'])
    rows = [t for c in chunks for t in c['test']]
    same = [t['rles'] == ref.get(t['stem'], []) for t in rows]
    print(f'test: rows identical to the submitted file on {sum(same)}/{len(same)} images ({[len(t["rles"]) for t in rows]} instances)')
    ok_test = all(same)
verdict = 'PASS' if all(exact) and ok_test is not False and all(c['tests']['returncode'] == 0 for c in chunks) else 'DIFFERENCES'
print('VERDICT:', verdict)
json.dump({'versions': chunks[0]['versions'], 'holdout_identical': sum(exact), 'holdout_pq': round(PQ(G.sum(0)), 4),
           'expected_pq': round(PQ(want.sum(0)), 4), 'test_identical': ok_test, 'verdict': verdict},
          open('analysis/results/e2e_summary.json', 'w'), indent=1)
