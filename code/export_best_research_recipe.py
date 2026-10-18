"""Export exact audited per-fold champion leaves without loading or fitting models."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/minimize_expanding_20261017'


def main():
    audit=json.loads((ROOT/'audit.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert audit['passed'] and audit['folds']==8
    folds=[]
    for folder in sorted((ROOT/'folds').iterdir()):
        if not folder.is_dir():continue
        meta=json.loads((folder/'manifest.json').read_text())
        selected=json.loads((folder/'selection.json').read_text())
        winners=selected['by_recipe']; recipe=winners['fixed_blend_ratio']
        rows={str(r['config']['id']):r for r in selected['candidates']}
        leaves=[]
        for name,weight in recipe['weights'].items():
            member=winners[name]
            for cid,seed_weight in member['components'].items():
                model=folder/'trials'/cid/'model.pt'
                assert model.is_file()
                leaves.append(dict(family=name,weight=weight*seed_weight,config=rows[cid]['config'],
                                   best_epoch=rows[cid]['best_epoch'],parameters=rows[cid]['parameters'],
                                   model_path=str(model.relative_to(PROJECT)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
        assert abs(sum(r['weight'] for r in leaves)-1)<1e-12 and min(r['weight'] for r in leaves)>0
        folds.append(dict(test_month=folder.name,train_start=meta['train_start'],train_end=meta['train_end'],
                          training_days=meta['train_days'],selection_month=meta['validation'],selection_days=[1,15],
                          calibration_days='16–month-end',recipe='fixed_blend_ratio',
                          family_weights=recipe['weights'],selection_ratio=recipe['selection_ratio'],leaves=leaves,
                          data_path=str((folder/'data.npz').relative_to(PROJECT)),
                          calibrated_predictions_path=str((folder/'selected/fixed_blend_ratio/predictions.npz').relative_to(PROJECT))))
    assert len(folds)==8
    output=PROJECT/'results/model_ranking_20261017/champion_per_fold.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(model_id='minimize_expanding_20261017/fixed_blend_ratio',
                                     source_sha256=manifest['source_sha256'],training_code_hashes=manifest['code_hashes'],
                                     quantiles=manifest['taus'],folds=folds,
                                     note='Exact evaluation checkpoints; no new all-history fit or test-driven weight changes.'),indent=2)+'\n')
    print(json.dumps(dict(path=str(output),folds=len(folds),checkpoint_leaves=sum(len(f['leaves']) for f in folds)),indent=2))


if __name__=='__main__':
    main()
