"""Quarter-aware numerical inputs; no pretrained weights or external sources."""
import numpy as np

def quarter_features(x):
    return np.concatenate([np.repeat(x,4,axis=0),np.tile(np.eye(4,dtype=np.float32),(len(x),1)),
        np.tile(np.arange(4,dtype=np.float32)/3,len(x))[:,None]],axis=1)

def cat_prediction(model,config,x):
    if config['strategy']=='pooled':
        return model.predict(quarter_features(x)).reshape(-1,4)
    return np.column_stack([m.predict(x) for m in model])

def constrain_residual(raw,config):
    if 'projection_weights' not in config:
        return raw-raw.mean(-1,keepdims=True)
    alpha=np.asarray(config['projection_weights'])
    assert alpha.shape==(4,4) and np.isfinite(alpha).all()
    assert np.max(np.abs(alpha.sum(1)-1))<1e-8 and (alpha>=0).all()
    assert len(raw)%96==0
    rows=np.tile(np.repeat(alpha,24,axis=0),(len(raw)//96,1))
    return raw-raw.sum(-1,keepdims=True)*rows

def learn_projection(raw,target,tr,va):
    """Fit frozen correction weights on validation; earlier/later-half safeguard."""
    from scipy.optimize import minimize
    mid=(tr+va)//2
    assert tr%96==0 and mid%96==0 and va%96==0
    def fit(r,t):
        uniform=np.full(4,.25)
        rr=r.reshape(-1,4);tt=t.reshape(-1,4);total=rr.sum(-1,keepdims=True)
        res=minimize(lambda a:np.abs(rr-total*a-tt).mean()+.05*np.sum((a-uniform)**2),uniform,
            method='SLSQP',bounds=[(0,1)]*4,constraints={'type':'eq','fun':lambda a:a.sum()-1},
            options={'maxiter':200,'ftol':1e-9})
        return res.x if res.success else uniform
    alpha=[];diagnostics=[]
    for z in range(4):
        r1=raw[tr:mid].reshape(-1,4,24,4)[:,z]
        t1=target[tr:mid].reshape(-1,4,24,4)[:,z]
        a=fit(r1,t1)
        r2=raw[mid:va].reshape(-1,4,24,4)[:,z]
        t2=target[mid:va].reshape(-1,4,24,4)[:,z]
        total=r2.sum(-1,keepdims=True)
        uniform_score=float(np.abs(r2-total/4-t2).mean())
        weighted_score=float(np.abs(r2-total*a-t2).mean())
        use=weighted_score<uniform_score-1e-5
        rr=raw[tr:va].reshape(-1,4,24,4)[:,z]
        tt=target[tr:va].reshape(-1,4,24,4)[:,z]
        alpha.append(fit(rr,tt) if use else np.full(4,.25))
        diagnostics.append(dict(zone=z,later_validation_half_uniform=uniform_score,
            later_validation_half_weighted=weighted_score,weighted_enabled=bool(use)))
    return np.asarray(alpha).tolist(),diagnostics
