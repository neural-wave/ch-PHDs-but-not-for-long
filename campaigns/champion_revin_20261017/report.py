"""Full-day plots comparing frozen champion against its fresh RevIN refit."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train as t

def main():
    assert json.loads((t.ROOT/'audit.json').read_text())['passed'];examples=[];rows=[]
    for normalized in [True,False]:
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,sharey=normalized)
        for i,(date,z) in enumerate([('2026-03-14',2),('2026-04-26',3)]):
            folder=t.ROOT/'folds'/date[:7];d=np.load(folder/'data.npz');start=sum(int(d[k]) for k in ['nt','ns','nc'])
            for j,recipe in enumerate(['original_champion','revin_champion']):
                v=np.load(folder/'selected'/recipe/'predictions.npz');ids=np.flatnonzero(v['dates']==date);assert len(ids)==1;k=int(ids[0]);q=v['quantiles'][k,z].reshape(96,11);y=v['observed'][k,z].ravel();lo=v['lower_90'][k,z].ravel();hi=v['upper_90'][k,z].ravel();h=d['h'][start+k,z];mu=h.mean();s=np.sqrt(h.var()+1);transform=(lambda a:(a-mu)/s) if normalized else (lambda a:a);x=(np.arange(96)+.5)/4;ax=axes[i,j]
                ax.fill_between(x,transform(lo),transform(hi),alpha=.5,color='#a4c5ed',label='Calibrated90%');ax.fill_between(x,transform(q[:,4]),transform(q[:,6]),alpha=.5,color='#4789ce',label='q25–q75')
                for n in [0,1,2,3,7,8,9,10]:ax.plot(x,transform(q[:,n]),color='#739bca',alpha=.35,lw=.45)
                ax.plot(x,transform(q[:,5]),color='#1359a0',lw=1.5,label='q50');ax.plot(x,transform(y),color='#d65b26',lw=1.1,label='Observed');mae=float(abs(q[:,5]-y).mean());width=float((hi-lo).mean());cover=float(((y>=lo)&(y<=hi)).mean());div=s if normalized else 1
                ax.set_title(f'{date} ZONE_{z} · {recipe}\nwidth90 {width/div:.3f} · MAE {mae/div:.3f} · coverage {cover:.1%}',fontsize=10);ax.grid(alpha=.2)
                if not normalized:
                    examples.append(dict(date=date,zone=z,recipe=recipe,width90=width,mae=mae,coverage90=cover,normalized_width90=width/s,normalized_mae=mae/s))
                    for hour in range(96):rows.append(dict(date=date,zone=z,recipe=recipe,hour=hour//4+1,quarter=hour%4+1,observed=y[hour],lower90=lo[hour],upper90=hi[hour],**{f'q{tau:g}':q[hour,n] for n,tau in enumerate(t.TAUS)}))
        axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Current champion versus SAME frozen ensemble recipe freshly fitted with RevIN\nPreviously inspected developmental examples; learned affine, all eleven quantiles');fig.supxlabel('Hour of delivery day');fig.supylabel('(price − known hourly daily mean) / input daily scale' if normalized else 'EUR/MWh');fig.tight_layout();name='full_day_normalized' if normalized else 'full_day_eur_mwh';fig.savefig(t.ROOT/(name+'.png'),dpi=170);fig.savefig(t.ROOT/(name+'.svg'));plt.close(fig)
    summary=pd.read_csv(t.ROOT/'summary.csv');pd.DataFrame(rows).to_csv(t.ROOT/'example_predictions.csv',index=False);(t.ROOT/'examples.json').write_text(json.dumps(examples,indent=2))
    lines=['# Frozen champion ensemble with RevIN','', 'Fresh RevIN fits of the51exact selected champion leaves across eight expanding monthly folds. Same architectures, lr/seeds, physical-currency ratio-weighted MAE+pinball objective and frozen blend weights. Stopping epochs reselected using prior days1–15; intervals calibrated on16–end. This is a frozen-recipe transfer ablation, not a new full candidate search or a production fit.','', summary.to_string(index=False),'', 'Original champion ratio must recover0.7079682867567635. Lower is better. All months are developmental. No deployment or registry promotion.','', 'Full-day panels compare original champion(left) with its fresh RevIN refit(right). full_day_normalized.png/svg has shared normalized axes; full_day_eur_mwh.png/svg shows original currency. Same March14ZONE2 and April26ZONE3 examples as the previous plot, fixed before this run. Raw quantile lines and calibrated90%bands are distinct. Examples are illustrative; no observed outcome was used to select new model weights.','',pd.DataFrame(examples).to_string(index=False),'', 'Source protocol/frozen_components.json and audit.json document input-only scales, learned-affine residual reversal, causality, exact weights, checkpoint replay, ordered quantiles, hourly conservation and CQR validation. Full predictions/checkpoints retained remotely in the recovery archive.']
    (t.ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT AND FULL-DAY PLOTS COMPLETE',flush=True)
if __name__=='__main__':main()
