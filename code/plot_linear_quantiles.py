"""Pinball/median learning curves and quantile calibration diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(out):
    out=Path(out);frame=pd.read_csv(out/'learning_curves.csv');interval=pd.read_csv(out/'interval_curves.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    old=Path('results/expanding_vast/learning_curves.csv')
    baseline=pd.read_csv(old) if old.exists() else None
    for plan in ['expanding','fixed_september']:
        data=frame[frame.plan==plan].sort_values('train_days')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        axes[0].plot(data.train_days,data.median_mae,'o-',color='#b43d33',label='Linear11-quantile median')
        if baseline is not None:
            for family in ['ridge','hierarchical','direct']:
                g=baseline[(baseline.plan==plan)&(baseline.family==family)&baseline.training_months.isin(data.training_months)].sort_values('train_days')
                label='Ridge baseline' if family=='ridge' else 'MAE-only '+family
                axes[0].plot(g.train_days,g.test_mae,'o--',label=label,alpha=.75)
        axes[1].plot(data.train_days,data.mean_pinball,'o-',color='#386d99')
        axes[0].set_ylabel('Median MAE (EUR/MWh)');axes[1].set_ylabel('Mean pinball loss (EUR/MWh)')
        for ax in axes:ax.set_xlabel('Training days');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(plan.replace('_',' ')+' — test metrics');fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(out/(plan+'_learning_curve.'+ext),dpi=150)
        plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for j,level in enumerate([.5,.9,.95]):
            for method,g in interval[(interval.plan==plan)&(interval.nominal==level)].groupby('method'):
                g=g.sort_values('train_days');axes[0,j].plot(g.train_days,g.coverage,'o-',label=method.replace('_',' '));axes[1,j].plot(g.train_days,g.mean_width,'o-')
            axes[0,j].axhline(level,color='black',ls=':',label='Nominal');axes[0,j].set_title(f'{round(level*100)}% interval');axes[0,j].set_ylim(0,1)
            axes[1,j].set_xlabel('Training days');axes[1,j].set_ylabel('Mean width (EUR/MWh)');axes[0,j].set_ylabel('Measured coverage')
        axes[0,0].legend(fontsize=7);fig.suptitle(plan.replace('_',' ')+' — interval quality');fig.tight_layout();fig.savefig(out/(plan+'_interval_quality.png'),dpi=150);plt.close(fig)
    report=json.loads((out/'metrics.json').read_text());last=report['folds'][-1]
    levels=report['quantiles'];fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],':',color='grey',label='Ideal calibration')
    ax.plot(levels,last['quantile_coverage'],'o-',label='All11 quantiles');ax.set(xlabel='Quantile level',ylabel='Observed fraction below prediction',title='Quantile calibration: '+last['plan']+' / '+last['test']);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'all11_quantile_calibration.png',dpi=150);plt.close(fig)
    assert (out/'fixed_september_learning_curve.png').stat().st_size>10000
