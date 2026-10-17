"""Finite-sample absolute-error split conformal; no fitting on calibration/test."""
import math
import numpy as np

def radius(scores,coverage):
    n=len(scores);rank=math.ceil((n+1)*coverage)
    if rank>n:return np.full(scores.shape[1:],np.inf),rank
    return np.sort(scores,axis=0)[rank-1],rank

def calibrate(cal_error,test_error):
    assert cal_error.ndim==4 and cal_error.shape[1:]==(4,24,4)
    results=[];bands={}
    for kind in ['marginal','hour_joint','day_joint']:
        scores=(cal_error.transpose(0,2,1,3).reshape(-1,4,4) if kind=='marginal' else
                cal_error.max(-1).transpose(0,2,1).reshape(-1,4) if kind=='hour_joint' else
                cal_error.max((1,2,3)))
        for coverage in [.5,.9,.95]:
            q,rank=radius(scores,coverage)
            rr=q[None,:,None,:] if kind=='marginal' else q[None,:,None,None] if kind=='hour_joint' else q.reshape(1,1,1,1)
            covered=test_error<=rr
            simultaneous=covered if kind=='marginal' else covered.all(-1) if kind=='hour_joint' else covered.all((1,2,3))
            finite=bool(np.isfinite(q).all());level=round(coverage*100)
            results.append(dict(kind=kind,nominal=coverage,coverage=float(simultaneous.mean()),quarter_coverage=float(covered.mean()),
                width=float(2*q.mean()) if finite else None,finite=finite,n=len(scores),rank=rank,
                radius=q.tolist() if finite else None,by_zone=covered.mean((0,2,3)).tolist(),
                daily_coverage=covered.mean((1,2,3)).tolist()))
            bands[f'{kind}_{level}']=rr
    return results,bands

def self_test():
    q,r=radius(np.arange(384),.9);assert r==347 and q==346
    q,r=radius(np.arange(384),.95);assert r==366 and q==365
    q,r=radius(np.arange(16),.9);assert r==16 and q==15
    q,r=radius(np.arange(16),.95);assert r==17 and np.isinf(q)
    rows,bands=calibrate(np.ones((16,4,24,4)),np.zeros((30,4,24,4)))
    assert len(rows)==9 and all(r['coverage']==1 for r in rows)
    assert all(np.diff([bands[f'{k}_{l}'].mean() for l in [50,90,95]]) .min()>=0 for k in ['marginal','hour_joint','day_joint'])

if __name__=='__main__':self_test();print('conformal rank/shape/coverage tests passed')
