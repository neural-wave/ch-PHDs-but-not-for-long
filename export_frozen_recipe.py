"""Export effective leaf weights/settings, without trained weights or external data."""
import json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
root=Path('results');sel=json.loads((root/'monthly_best.json').read_text());src=root/sel['best_directory']
with (src/'model.pkl').open('rb') as f:b=pickle.load(f)
leaves={}
def walk(obj,w):
    for item in obj['models']:
        ww=w*np.asarray(item['weight'])
        if item['name']=='parent_ensemble':walk(item['artifact'],ww);continue
        key=(item['name'],json.dumps(item['config'],sort_keys=True))
        if key not in leaves:leaves[key]=dict(name=item['name'],config=item['config'],weight=np.zeros(4))
        leaves[key]['weight']+=ww
walk(b,np.ones(4))
models=[dict(x,weight=x['weight'].tolist()) for x in leaves.values()]
assert np.allclose(np.asarray([x['weight'] for x in models]).sum(0),1)
recipe=dict(version=1,selection_basis='August validation only',source_evaluation_directory=src.name,
    source_evaluation_model_sha256=json.loads((src/'audit.json').read_text())['artifact_sha256'],
    source_sha256=b['source_sha256'],feature_mode='rich',model_fit_days=363,normalization='all363 training days',
    tf32=False,models=models,description='Frozen effective leaf weights; flattened nested convex mixtures algebraically. Retrain learned components from scratch on official training only.')
Path('frozen_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print('EXPORTED',len(models),'leaf settings')
