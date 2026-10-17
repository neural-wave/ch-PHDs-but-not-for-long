from pathlib import Path
import numpy as np,pandas as pd
PROJECT=Path(__file__).resolve().parents[2]
root=PROJECT/'results/minimize_expanding_20261017';rows=[]
for folder in sorted((root/'folds').iterdir()):
 z=np.load(folder/'selected/fixed_blend_ratio/predictions.npz');q=z['quantiles'];y=z['observed'];w=(z['upper_90']-z['lower_90']).mean(-1);rough=abs(np.diff(y,n=2,axis=-1)).mean(-1)
 for i,date in enumerate(z['dates']):
  for zone in range(4):
   cov=((y[i,zone]>=z['lower_90'][i,zone])&(y[i,zone]<=z['upper_90'][i,zone])).mean()
   if cov<.90:continue
   rw=np.convolve(w[i,zone],np.ones(3)/3,'valid');rr=np.convolve(rough[i,zone],np.ones(3)/3,'valid')
   calm=int(rw.argmin());vol=int(rw.argmax());ratio=rw[vol]/rw[calm]
   if rr[calm]>2 or rr[vol]<8 or rw[calm]>12 or abs(vol-calm)<3:continue
   rows.append(dict(date=str(date),zone=zone,calm=calm,volatile=vol,calm_width=rw[calm],volatile_width=rw[vol],width_ratio=ratio,calm_rough=rr[calm],volatile_rough=rr[vol],coverage=cov))
df=pd.DataFrame(rows).sort_values('width_ratio',ascending=False);print(df.head(15).to_string(index=False));out=PROJECT/'results/illustrative_quantile_bands_20261017';df.to_csv(out/'stronger_contrast_candidates.csv',index=False)
