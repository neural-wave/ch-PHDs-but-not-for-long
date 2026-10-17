"""Numerical hourly constraint and validation/holdout isolation checks."""
import numpy as np
from catboost_models import constrain_residual,learn_projection

rng=np.random.default_rng(42)
raw=rng.normal(size=(384,4))*20
uniform=constrain_residual(raw,{})
alpha=np.tile([.4,.3,.2,.1],(4,1)).tolist()
weighted=constrain_residual(raw,{'projection_weights':alpha})
assert np.max(np.abs(uniform.sum(1)))<1e-12
assert np.max(np.abs(weighted.sum(1)))<1e-12
assert np.allclose(uniform,constrain_residual(raw,{'projection_weights':np.full((4,4),.25).tolist()}))
target=weighted+rng.normal(size=weighted.shape)*.1;target-=target.mean(1,keepdims=True)
a,diag=learn_projection(raw,target,96,288)
changed=target.copy();changed[288:]=changed[288:]*100+1234
aa,dd=learn_projection(raw,changed,96,288)
assert np.array_equal(a,aa) and diag==dd
assert np.max(np.abs(constrain_residual(raw,{'projection_weights':a}).sum(1)))<1e-10
print('PASS: hourly constraint, uniform equivalence, and holdout-target isolation',flush=True)
