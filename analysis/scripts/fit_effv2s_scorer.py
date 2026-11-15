"""Store the instance scorer of the final (EfficientNetV2-S) submission as a file: models/effv2s_scorer.{txt,json}.

The 0.38 submission (analysis/scripts/make_effv2s_submission.py) refit the scorer in memory. This script repeats that
fit exactly (near-limb out-of-fold instances of the 657 pool images, released LightGBM parameters, 36 features,
tau = PQ(near-limb out-of-fold) / 2), writes the model as text, reloads it the way the pipeline does
(filament_scorer.InstanceScorer.load(name="effv2s_scorer")) and checks that the reloaded model rebuilds the submitted
CSV byte for byte from the stored test instances (sha256 below; the CSV itself is in the weights dataset).
"""
import csv
import hashlib
import io
import json
import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(0, 'analysis/scripts')
sys.path.insert(0, '.')
import filament_scorer as fs                                                  # noqa: E402
from scorer_common import PARAMS, PQ, counts_after, feature_names, fit, table   # noqa: E402

SUBMITTED_SHA256 = '031f80ba8f719232b99ff76e9bc226ee0a8e4cf1d77607ca19880e5d9a29d520'   # submission 58333664
LIMB_T = 0.88


def submission_bytes(scorer, images):
    """The submission CSV built from stored test instances ({'stem', 'instances': [{'rle', 'feats'}]})."""
    rows = []
    for im in images:
        inst = [it for it in im['instances'] if it['feats']['dist_r'] <= LIMB_T]
        kept = [it['rle'] for it, v in zip(inst, scorer.scores(inst)) if v >= scorer.tau]
        rows += [{'filament_id': f"{im['stem']}_{i}", 'segmentation_rle': c} for i, c in enumerate(kept, 1)]
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=['filament_id', 'segmentation_rle'])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode('utf-8'), len(rows)


if __name__ == '__main__':
    new = json.load(open('analysis/results/effv2s/effv2s_instances.json'))
    assert new['status'] == 'done' and new['folds'] == [1, 2, 3, 4, 5]
    oof = sorted((r for r in new['results'] if r['set'] == 'oof'), key=lambda r: r['name'])
    assert len(oof) == 657
    feats = feature_names(oof)
    assert len(feats) == 36
    Dall = table(oof, feats)
    inner = np.array([it['feats']['dist_r'] <= LIMB_T for r in oof for it in r['instances']])
    pq_nl = PQ(counts_after(Dall, inner).sum(0))
    tau = pq_nl / 2
    Dn = table([{**r, 'instances': [it for it in r['instances'] if it['feats']['dist_r'] <= LIMB_T]} for r in oof], feats)
    model = fit(Dn)
    model.booster_.save_model('models/effv2s_scorer.txt')
    json.dump({'features': feats, 'tau': tau, 'limb_drop': LIMB_T,
               'fitted_on': f"{len(Dn['y'])} near-limb out-of-fold instances of the EfficientNetV2-S stage-A, 657 images",
               'tau_rule': f"near-limb out-of-fold PQ {pq_nl:.4f} / 2", 'lightgbm': lgb.__version__, 'params': PARAMS},
              open('models/effv2s_scorer.json', 'w'), indent=1)
    print(f"fitted on {len(Dn['y'])} instances, near-limb out-of-fold PQ {pq_nl:.4f}, tau {tau:.6f}")

    sc = fs.InstanceScorer.load(name='effv2s_scorer')
    test = json.load(open('analysis/results/effv2s/effv2s_test_instances.json'))
    assert test['status'] == 'done' and len(test['images']) == 180
    data, n = submission_bytes(sc, test['images'])
    sha = hashlib.sha256(data).hexdigest()
    print(f"stored model: {n} rows, sha256 {sha}")
    assert sha == SUBMITTED_SHA256, 'the stored model does not rebuild the submitted CSV'
    print('OK: models/effv2s_scorer.txt rebuilds the submitted CSV byte for byte')
