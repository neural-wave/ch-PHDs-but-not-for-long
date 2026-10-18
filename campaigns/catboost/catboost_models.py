"""Quarter-aware numerical inputs; no pretrained weights or external sources."""
import numpy as np

def quarter_features(x):
    return np.concatenate([np.repeat(x,4,axis=0),np.tile(np.eye(4,dtype=np.float32),(len(x),1)),
        np.tile(np.arange(4,dtype=np.float32)/3,len(x))[:,None]],axis=1)

def cat_prediction(model,config,x):
    if config['strategy']=='pooled':
        return model.predict(quarter_features(x)).reshape(-1,4)
    return np.column_stack([m.predict(x) for m in model])
