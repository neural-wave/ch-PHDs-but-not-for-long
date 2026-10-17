"""Recompute metrics and training statistics from saved evidence."""
import json
import numpy as np
import torch
from joint_zone_cnn import ROOT, JointCNN, inputs, configs

d=np.load(ROOT/'data.npz');report=json.loads((ROOT/'metrics.json').read_text())
selection=json.loads((ROOT/'selection.json').read_text());rows=report['candidates']
assert len(rows)==len(configs()) and len({r['config']['id'] for r in rows})==len(configs())
audits={}
for key,winner in selection.items():
    eligible=[r for r in rows if int(r['config']['history'])==int(key)]
    assert winner['config']['id']==min(eligible,key=lambda r:r['validation_mae'])['config']['id']
    f=np.load(ROOT/f'features_{key}.npz');raw=inputs(d['dates'],d['h'],d['dates'],d['y'],bool(int(key)))
    np.testing.assert_array_equal(raw,f['raw'])
    np.testing.assert_allclose(f['mu'],raw[:302].mean((0,2),keepdims=True))
    np.testing.assert_allclose(f['sd'],raw[:302].std((0,2),keepdims=True).clip(.05))
    folder=ROOT/'trials'/str(winner['config']['id']);pred=np.load(folder/'predictions.npy')
    val=float(np.abs(pred[:31]-d['y'][302:333]).mean());test=float(np.abs(pred[31:]-d['y'][333:]).mean())
    assert abs(val-winner['validation_mae'])<1e-4
    assert abs(test-report['by_history'][key]['september_mae'])<1e-10
    assert np.abs(pred.mean(-1)-d['h'][302:]).max()<1e-8
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    net=JointCNN(f['x'].shape[1],winner['config']).cuda().eval();net.load_state_dict(checkpoint['state'])
    with torch.no_grad():rr=net(torch.tensor(f['x'][302:],device='cuda')).cpu().numpy().astype('float64')
    reconstructed=d['b'][302:]+20*(rr-rr.mean(-1,keepdims=True))
    np.testing.assert_allclose(pred,reconstructed,atol=1e-7,rtol=0)
    audits[key]=dict(validation_mae=val,september_mae=test,by_zone=np.abs(pred[31:]-d['y'][333:]).mean((0,2,3)).tolist())
(ROOT/'audit.json').write_text(json.dumps(dict(passed=True,by_history=audits,checks=[f'{len(rows)} fits','August selection','train-only normalization','feature reconstruction','hourly conservation','checkpoint replay']),indent=2))
text=['# Joint four-zone CNN comparison','', 'August selects; September is developmental. No deployment refit.','',
      '| History | August MAE | September MAE |','|---|---:|---:|']
for key,r in audits.items():text.append(f"| {'Previous 7 days, hourly and quarter-hour' if key=='1' else 'None'} | {r['validation_mae']:.6f} | {r['september_mae']:.6f} |")
(ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
print('AUDIT PASSED',audits,flush=True)
