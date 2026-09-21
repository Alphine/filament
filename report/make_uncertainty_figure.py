"""Figure for the technical report: PQ with 95% bootstrap intervals, model versus annotator agreement.

Run from the repository root:  python report/make_uncertainty_figure.py
Inputs : analysis/results/fpdiag/*.json (per-image official pair counts of the final pipeline),
         analysis/results/annotator_agreement_summary.json (annotator-vs-annotator PQ).
Output : report/overleaf/fig_uncertainty.pdf and report/uncertainty_numbers.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
imgs = []
for i in range(4):
    imgs += json.load(open(f'analysis/results/fpdiag/holdout_chunk{i}.json'))['results']


def stats(r):
    S = tp = fp = fn = 0
    for p in r['passes']:
        hp = {x[0] for x in p['pairs']}
        hg = {x[1] for x in p['pairs']}
        S += sum(x[2] for x in p['pairs'])
        tp += len(p['pairs'])
        fp += p['n_pred'] - len(hp)
        fn += p['n_gt'] - len(hg)
    return S, tp, fp, fn


def PQ(t):
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


per = np.array([stats(r) for r in imgs])
n_pass = np.array([len(r['passes']) for r in imgs])


def boot(mask, n=4000):
    sub = per[mask]
    vals = [PQ(sub[rng.integers(0, len(sub), len(sub))].sum(0)) for _ in range(n)]
    return PQ(sub.sum(0)), np.percentile(vals, [2.5, 97.5])


agree = json.load(open('analysis/results/annotator_agreement_summary.json'))
h50, allmp = agree['holdout50'], agree['all_multipass']
z = 1.96
bars = [
    ('Model\nall 50', *boot(np.ones(len(per), bool))),
    ('Model\n19 multi-pass', *boot(n_pass >= 2)),
    ('Model\n31 single-pass', *boot(n_pass == 1)),
    ('Annotator vs.\nannotator\n(same 19)', h50['pq'], (h50['pq'] - z * h50['pq_se_bootstrap'], h50['pq'] + z * h50['pq_se_bootstrap'])),
    ('Annotator vs.\nannotator\n(296 train)', allmp['pq'], (allmp['pq'] - z * allmp['pq_se_bootstrap'], allmp['pq'] + z * allmp['pq_se_bootstrap'])),
]
vals = np.array([b[1] for b in bars])
lo = np.array([b[1] - b[2][0] for b in bars])
hi = np.array([b[2][1] - b[1] for b in bars])
colors = ['#2b6cb0', '#2b6cb0', '#2b6cb0', '#c05621', '#c05621']

fig, ax = plt.subplots(figsize=(3.4, 2.15))
x = np.arange(len(bars))
ax.bar(x, vals, color=colors, width=0.62)
ax.errorbar(x, vals, yerr=[lo, hi], fmt='none', ecolor='black', capsize=3, lw=1)
for xi, v in zip(x, vals):
    ax.text(xi, 0.015, f'{v:.2f}', ha='center', va='bottom', color='white', fontsize=7, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=5.6)
ax.set_ylabel('PQ (95% interval)', fontsize=7)
ax.set_ylim(0, 0.62)
ax.tick_params(axis='y', labelsize=6.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout(pad=0.3)
os.makedirs('report/overleaf', exist_ok=True)
plt.savefig('report/overleaf/fig_uncertainty.pdf')
json.dump({b[0].replace('\n', ' '): {'pq': round(float(b[1]), 4), 'ci95': [round(float(b[2][0]), 4), round(float(b[2][1]), 4)]} for b in bars},
          open('report/uncertainty_numbers.json', 'w'), indent=1)
for b in bars:
    print(f"{b[0].replace(chr(10), ' '):38s} PQ {b[1]:.4f}  CI95 [{b[2][0]:.3f}, {b[2][1]:.3f}]")
