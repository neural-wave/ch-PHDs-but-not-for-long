"""Causal invariance and missing-calendar-date checks for feature implementation."""
import numpy as np
from reconstruction import read_targets,features,rich_features

dates,y=read_targets('train_prices_15min.csv')
probe=250; h=y.mean(-1)
ds=dates[probe:probe+2]; hs=h[probe:probe+2]
x=rich_features(ds,hs,dates,y)
assert np.isfinite(x).all() and x.shape[0]==192
basic=features(ds,hs,dates,y)
assert np.array_equal(x[:,:basic.shape[1]],basic)
# Poisoning both current/future targets changes nothing for the first query day.
poison=y.copy(); poison[probe:]=poison[probe:]*47-891
assert np.array_equal(x[:96],rich_features(ds[:1],hs[:1],dates,poison))
# Removing all unavailable future/current records is equivalent to filtering inside.
assert np.array_equal(x[:96],rich_features(ds[:1],hs[:1],dates[:probe],y[:probe]))
# Query with no history must still be finite and retain the same feature dimension.
empty=rich_features(ds[:1],hs[:1],dates[:0],y[:0])
assert empty.shape==x[:96].shape and np.isfinite(empty).all()
print({'status':'passed','rows':len(x),'basic_features':basic.shape[1],'rich_features':x.shape[1]},flush=True)
