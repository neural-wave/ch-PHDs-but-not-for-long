from pathlib import Path
import numpy as np,pandas as pd
PROJECT=Path(__file__).resolve().parents[2]
root=PROJECT/'results/minimize_expanding_20261017';rows=[]
for folder in sorted((root/'folds').iterdir()):
 z=np.load(folder/'selected/fixed_blend_ratio/predictions.npz');y=z['observed'];q=z['quantiles'];h=y.mean(-1);s=np.sqrt(h.var(-1)+1);w=z['upper_90']-z['lower_90'];pr=abs(np.diff(q[...,5],n=2,axis=-1)).mean(-1)/s[:,:,None];yr=abs(np.diff(y,n=2,axis=-1)).mean(-1)/s[:,:,None];ww=w.mean(-1)/s[:,:,None]
 for i,date in enumerate(z['dates']):
  for zone in range(4):
   cor=pd.Series(ww[i,zone]).rank().corr(pd.Series(yr[i,zone]).rank())
   rows.append(dict(date=str(date),zone=zone,scale=s[i,zone],width=float(ww[i,zone].mean()),rough=float(yr[i,zone].mean()),predrough=float(pr[i,zone].mean()),coverage=float(((y[i,zone]>=z['lower_90'][i,zone])&(y[i,zone]<=z['upper_90'][i,zone])).mean()),corr=cor,mae=float(abs(q[i,zone,...,5]-y[i,zone]).mean()/s[i,zone])))
df=pd.DataFrame(rows);out=PROJECT/'results/illustrative_quantile_bands_20261017';out.mkdir(exist_ok=True);df.to_csv(out/'candidates.csv',index=False)
rough=df[(df.coverage>=.9)&(df['corr']>.35)&(df.rough>.25)&(df.predrough>.08)&(df.scale>10)&(df.scale<80)].copy();rough['score']=rough.rough*rough['corr'];print('JAGGED CANDIDATES');print(rough.sort_values('score',ascending=False).head(12).to_string(index=False))
print('SMOOTH CANDIDATES');print(df[(df.coverage>=.94)&(df.rough<.10)&(df.scale>10)&(df.scale<80)].sort_values('width').head(12).to_string(index=False))
