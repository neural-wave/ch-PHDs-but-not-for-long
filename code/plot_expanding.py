"""Standalone scientific learning curves generated on the execution host."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(out):
    out=Path(out);frame=pd.read_csv(out/'learning_curves.csv')
    colors=dict(hierarchical='#bd3e36',direct='#3576b5',ridge='#398354',interpolation='#8a8a8a')
    labels=dict(hierarchical='Hierarchical neural',direct='Direct neural',ridge='Linear ridge',interpolation='Interpolation')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    for plan,title in [('expanding','Expanding training: each point tests a different month'),('fixed_september','More training history: same August validation and September test')]:
        data=frame[frame.plan==plan]
        fig,ax=plt.subplots(figsize=(9,5))
        for family,g in data.groupby('family'):
            g=g.sort_values('train_days');ax.plot(g.train_days,g.test_mae,'o-',label=labels[family],color=colors[family])
        for _,r in data[data.family=='hierarchical'].iterrows():
            ax.annotate(r.test,(r.train_days,r.test_mae),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xlabel='Training days',ylabel='Test MAE (EUR/MWh; lower is better)',title=title)
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        for suffix in ['png','svg']:fig.savefig(out/f'{plan}_learning_curve.{suffix}')
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for z,ax in enumerate(axes.flat):
            for family,g in data.groupby('family'):
                g=g.sort_values('train_days');ax.plot(g.train_days,g[f'mae_ZONE_{z}'],'o-',label=labels[family],color=colors[family])
            ax.set_title(f'ZONE_{z}');ax.set_xlabel('Training days');ax.set_ylabel('Test MAE');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle(title);fig.tight_layout()
        fig.savefig(out/f'{plan}_by_zone.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for plan,ax in zip(['expanding','fixed_september'],axes):
        data=frame[frame.plan==plan].pivot(index='train_days',columns='family',values='test_mae').sort_index()
        delta=data['hierarchical']-data['direct'];ax.axhline(0,color='grey',lw=1);ax.plot(delta.index,delta,'o-',color=colors['hierarchical'])
        ax.set(xlabel='Training days',ylabel='Hierarchy MAE − direct MAE',title=plan.replace('_',' '));ax.grid(alpha=.2)
    fig.suptitle('Hierarchy benefit: below zero favors hierarchy');fig.tight_layout();fig.savefig(out/'hierarchy_benefit.png');plt.close(fig)
    stage=out/'fixed_september_10m'/'hierarchical_stages.npz'
    if stage.exists():
        with np.load(stage) as s:pred=s['predicted'][0,0];truth=s['observed'][0,0];date=str(s['dates'][0])
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for i,(ax,name) in enumerate(zip(axes,['Half-hour contrast a','First half-hour quarter contrast u','Second half-hour quarter contrast v'])):
            ax.plot(range(1,25),truth[:,i],label='Observed',color='black');ax.plot(range(1,25),pred[:,i],label='Predicted',color=colors['hierarchical']);ax.set_ylabel('EUR/MWh');ax.set_title(name);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Hour');fig.suptitle(f'Interpretable output stages: ZONE_0, {date}');fig.tight_layout();fig.savefig(out/'hierarchy_stages_example.png');plt.close(fig)
    assert (out/'fixed_september_learning_curve.png').stat().st_size>10000


if __name__=='__main__':
    import sys
    plot(sys.argv[1])
