"""RevIN refit of the exact frozen champion leaves and blend weights."""
import sys,hashlib,json,argparse,concurrent.futures as cf,multiprocessing as mp,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'campaigns/monthly_quantiles_20261017'))
import trainq as e
import numpy as np
import pandas as pd
import torch
ROOT=PROJECT/'results/champion_revin_20261017';SOURCE=PROJECT/'results/minimize_expanding_20261017'
e.ROOT=e.p.ROOT=ROOT
TAUS=e.TAUS

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def registry():return json.loads((HERE/'frozen_components.json').read_text())
def config(leaf):return dict(leaf['config'],preprocessing='revin_champion',quantile_weight=1.)

def query_features(date,h,hd,hy,raw=None):
    """Current hourly statistics transform both current and strictly past inputs."""
    mu=h.mean(-1);s=np.sqrt(h.var(-1)+1)
    hh=20*(h-mu[...,None])/s[...,None]
    prior=20*(hy-mu[0,:,None,None])/s[0,:,None,None]
    f=e.p.features(np.asarray(date,dtype=object),hh,np.asarray(hd,dtype=object),prior)
    # Preserve original historical-analog selection/weights, normalizing outputs.
    if raw is not None:
        f['cnn'][:,-4:]=raw['cnn'][:,-4:]*20/s[:,:,None]
        f['analog']=raw['analog']*20/s[:,:,None,None]
        f['compact'][...,90:94]=raw['compact'][...,90:94]*20/s[:,:,None,None]
        f['rich'][...,377:381]=raw['rich'][...,377:381]*20/s[:,:,None,None]
    # Original absolute mean/std side information is retained where available.
    f['compact'][...,3:5]=np.stack([mu/100,h.std(-1)/100],-1)[:,:,None,:]
    f['rich'][...,339:341]=np.stack([mu/100,h.std(-1)/100],-1)[:,:,None,:]
    return f

def prepare():
    e.p.setup();ROOT.mkdir(parents=True,exist_ok=True);reg=registry()
    assert sha(PROJECT/'train_prices_15min.csv')==e.p.SHA==reg['source_sha256']
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    for path,value in reg['training_code_hashes'].items():assert sha(PROJECT/path)==value,path
    raw=np.load(SOURCE/'raw.npz');dates=raw['dates'];features={k:[] for k in ['rich','compact','cnn','curve','analog']}
    for i in range(len(dates)):
        source={k:raw[k][i:i+1] for k in features}
        f=query_features(dates[i:i+1],raw['h'][i:i+1],dates[:i].astype(object),raw['y'][:i],source)
        for k in features:features[k].append(f[k])
        if i%60==0:print('NORMALIZED FEATURES',i,len(dates),flush=True)
    arrays={k:raw[k] for k in ['dates','y','h','b']};arrays.update({k:np.concatenate(v) for k,v in features.items()});arrays['scale']=np.sqrt(raw['h'].var(-1)+1)
    assert arrays['rich'].shape[-1]==389 and arrays['compact'].shape[-1]==190 and arrays['cnn'].shape[1]==175
    # Full-history versus truncated/perturbed future must be identical.
    for i in [0,91,150,302,333]:
        changed=raw['y'].copy();changed[i:]+=np.arange(4)*123456
        source={k:raw[k][i:i+1] for k in features}
        f=query_features(dates[i:i+1],raw['h'][i:i+1],dates.astype(object),changed,source)
        for k in features:np.testing.assert_allclose(f[k],arrays[k][i:i+1],rtol=0,atol=2e-6)
    np.savez(ROOT/'raw.npz',**arrays)
    paths=list(reg['training_code_hashes'])+[str(p.relative_to(PROJECT)) for p in HERE.iterdir() if p.suffix in ['.py','.md','.json']]
    manifest=dict(source_sha256=e.p.SHA,source_model=reg['model_id'],frozen_components_sha256=sha(HERE/'frozen_components.json'),code_hashes={p:sha(PROJECT/p) for p in paths},feature_source_sha256=sha(ROOT/'raw.npz'),source_raw_sha256=sha(SOURCE/'raw.npz'),folds=reg['folds'],fresh_fits=51,causality_passed=True)
    e.p.write(ROOT/'manifest.json',manifest);print('PREPARED 51 EXACT CHAMPION LEAF REFITS',flush=True)

def fold_data(fold,smoke=False):
    src=SOURCE/'folds'/fold['test_month'];d=dict(np.load(src/'data.npz'));raw=np.load(ROOT/'raw.npz');n=len(d['dates']);nt=int(d['nt'])
    for k in ['rich','compact','cnn','curve']:
        x=np.clip(raw[k][:n],-15,15).astype('float32')
        if k=='compact':x=np.concatenate([x,np.broadcast_to(np.eye(4)[None,:,None,:],(n,4,24,4))],-1).astype('float32')
        d[k]=x;d[k+'_mu']=np.zeros_like(d[k+'_mu']);d[k+'_sd']=np.ones_like(d[k+'_sd'])
        if k in ['rich','compact']:d[k+'_knots']=np.quantile(x[:nt].reshape(-1,x.shape[-1]),[.25,.5,.75],axis=0).astype('float32')
    d['scale']=raw['scale'][:n];folder=ROOT/('smoke' if smoke else 'folds')/fold['test_month'];folder.mkdir(parents=True,exist_ok=True);np.savez(folder/'data.npz',**d)
    meta=json.loads((src/'manifest.json').read_text());meta.update(preprocessing='input-day RevIN; shared learned affine; exact champion leaves/weights frozen')
    e.p.write(folder/'manifest.json',meta);return folder

class Quantiles(e.Quantiles):
    def __init__(self,cfg,data):
        super().__init__(cfg,data)
        self.log_gamma=torch.nn.Parameter(torch.zeros(()));self.beta=torch.nn.Parameter(torch.zeros(()))
        # Monetary channel masks; do not transform clocks, masks or side stats.
        n=data[self.feature].shape[1] if self.two_d else data[self.feature].shape[-1]
        price=np.zeros(n);level=np.zeros(n)
        if self.feature=='rich':
            price[:248]=1;price[287:339]=1;price[341:381]=1;price[385:389]=1
            level[:96]=.2;level[192:220]=.2;level[341:345]=.2
        elif self.feature=='compact':
            price[[0,1,2]]=1;price[5:9]=1;price[18:190]=1
            price[np.arange(35,90,9)]=0
            level[0]=.2;level[94:190]=.2
        else:
            price[:64]=1;price[80:156]=1;price[171:175]=1
            level[:64]=.2;level[144:148]=.2
        self.register_buffer('price_mask',torch.tensor(price,dtype=torch.float32));self.register_buffer('level_mask',torch.tensor(level,dtype=torch.float32))
    def affine(self,x,curve):
        gamma=self.log_gamma.clamp(-3,3).exp()
        if self.two_d:
            price=self.price_mask[None,:,None];level=self.level_mask[None,:,None].expand_as(x).clone()
            # Unavailable historical prices remain zero under learned centering.
            level[:,:64]*=x[:,64:80].repeat(1,4,1)
        else:price=self.price_mask;level=self.level_mask
        x=x*(1+(gamma-1)*price)+self.beta*level
        return x,curve*gamma+.2*self.beta,gamma
    def forward(self,x,curve):
        x,curve,gamma=self.affine(x,curve)
        return super().forward(x,curve)/gamma

def physical(raw,b,s):
    raw=np.asarray(raw,dtype='float64');raw-=raw[...,5].mean(-1,keepdims=True)[...,None]
    out=b[...,None]+s[:,:,None,None,None]*raw
    assert np.isfinite(out).all() and np.diff(out,axis=-1).min()>=-1e-8
    return out

def predict(net,d,nt):
    chunks=[]
    with torch.no_grad():
        for i in range(nt,len(d['y']),32):chunks.append(net(torch.tensor(d[net.feature][i:i+32],device='cuda'),torch.tensor(d['curve'][i:i+32],device='cuda')).cpu().numpy())
    return physical(np.concatenate(chunks),d['b'][nt:],d['scale'][nt:])

def fit(leaf,folder,smoke=False):
    cfg=config(leaf);e.p.setup(cfg['seed']);folder=Path(folder);d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns']);started=time.time()
    net=Quantiles(cfg,d);net.initialize((d['y'][:nt]-d['b'][:nt])/d['scale'][:nt,:,None,None]);net.cuda()
    x=torch.tensor(d[net.feature][:nt+ns],device='cuda');curve=torch.tensor(d['curve'][:nt+ns],device='cuda');target=torch.tensor((d['y'][:nt+ns]-d['b'][:nt+ns])/20,dtype=torch.float32,device='cuda');mult=torch.tensor(d['scale'][:nt+ns,:,None,None,None]/20,dtype=torch.float32,device='cuda');weights=torch.tensor(d['weights_ratio'],device='cuda');tau=torch.tensor(TAUS,dtype=torch.float32,device='cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01);best=float('inf');be=0;state=None;curves=[]
    for epoch in range(1,(2 if smoke else 450 if net.two_d else 300)+1):
        net.train()
        for ids in torch.randperm(nt,device='cuda').split(32):
            err=target[ids,...,None]-net(x[ids],curve[ids])*mult[ids]
            loss=((err[...,5].abs().mean((1,2,3))+torch.maximum(tau*err,(tau-1)*err).mean((1,2,3,4)))*weights[ids]).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():
            err=target[nt:,...,None]-net(x[nt:],curve[nt:])*mult[nt:];val=float(err[...,5].abs().mean())*20;pb=float(torch.maximum(tau*err,(tau-1)*err).mean())*20
        assert np.isfinite(val+pb);curves.append([epoch,val/float(d['selection_interpolation_mae']),pb])
        if val<best:best=val;be=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-be>=(50 if net.two_d else 35):break
    net.load_state_dict(state);net.eval();q=predict(net,d,nt);dest=folder/'trials'/str(cfg['id']);dest.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=cfg,state=state,feature=net.feature),dest/'model.pt');np.save(dest/'quantiles.npy',q);np.savetxt(dest/'curves.csv',curves,delimiter=',',header='epoch,selection_ratio,pinball',comments='')
    val=float(abs(q[:ns,...,5]-d['y'][nt:nt+ns]).mean());row=dict(config=cfg,weight=leaf['weight'],best_epoch=be,epochs=epoch,validation_mae=val,selection_ratio=val/float(d['selection_interpolation_mae']),parameters=sum(p.numel() for p in net.parameters()),seconds=time.time()-started);e.p.write(dest/'metrics.json',row);return row

def select(folder,fold,rows,smoke=False):
    d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns']);weights={str(v['config']['id']):v['weight'] for v in fold['leaves'] if any(r['config']['id']==v['config']['id'] for r in rows)}
    # Smoke uses a representative subset only; full run exactly preserves weights.
    total=sum(weights.values());weights={k:v/total for k,v in weights.items()};q=sum(np.load(folder/'trials'/k/'quantiles.npy')*v for k,v in weights.items());dest=folder/'trials/revin_champion';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',q)
    winners={}
    for name,values in [('revin_champion',q),('original_champion',np.load(SOURCE/'folds'/folder.name/'trials/fixed_blend_ratio/quantiles.npy'))]:
        dest=folder/'trials'/name;dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',values);mae=float(abs(values[:ns,...,5]-d['y'][nt:nt+ns]).mean())
        winners[name]=dict(config=dict(id=name,family='frozen_champion_blend',objective='ratio'),selection_ratio=mae/float(d['selection_interpolation_mae']),validation_mae=mae,weights=weights if name=='revin_champion' else fold['family_weights'])
    empirical=np.quantile(d['y'][:nt]-d['b'][:nt],TAUS,axis=(0,2)).transpose(1,2,0);empirical[...,:5]=np.minimum(empirical[...,:5],0);empirical[...,5]=0;empirical[...,6:]=np.maximum(empirical[...,6:],0)
    dest=folder/'trials/interpolation';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',d['b'][nt:,...,None]+empirical[None,:,None,:,:]);winners['interpolation']=dict(config=dict(id='interpolation',family='interpolation'),selection_ratio=1.,validation_mae=float(d['selection_interpolation_mae']))
    # Fixed requested output, not a new test-selected recipe.
    winners['validation_selector']=dict(winners['revin_champion'],chosen_recipe='revin_champion')
    e.p.write(folder/'selection.json',dict(by_recipe=winners,candidates=rows,frozen_weights=weights));return winners

def run(smoke=False):
    folds=registry()['folds'];reports=[]
    if smoke:folds=[folds[0],folds[-1]]
    for fold in folds:
        folder=fold_data(fold,smoke);leaves=fold['leaves']
        if smoke:
            seen=set();leaves=[v for v in leaves if not(v['config']['family'] in seen or seen.add(v['config']['family']))]
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for f in cf.as_completed([pool.submit(fit,v,str(folder),smoke) for v in leaves]):
                row=f.result();rows.append(row);print('FIT',fold['test_month'],len(rows),len(leaves),row,flush=True)
        report=e.evaluate(folder,select(folder,fold,rows,smoke));reports.append(report);e.summarize(reports,smoke);e.p.write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_folds=len(reports),total_folds=len(folds),last_test_month=fold['test_month']))
    if not smoke:pd.concat([pd.read_csv(ROOT/'folds'/f['test_month']/'predictions.csv.gz') for f in folds],ignore_index=True).to_csv(ROOT/'predictions.csv.gz',index=False,compression='gzip')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','run']);a=p.parse_args();prepare() if a.stage=='prepare' else run(a.stage=='smoke')
