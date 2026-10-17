"""Exact point-model linearity, modal frequencies, effective kernels and contributions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train as t
NAMES=[f'ZONE_{z}_hourly' for z in range(4)]+[f'ZONE_{z}_past_interpolation_error' for z in range(4)]+['hourly_available','quarters_observed']


def smoke():
    t.setup(17);checks={}
    for family in t.FAMILIES:
        cfg=next(c for c in t.configs() if c['family']==family);net=t.QuantileSSM(cfg,1,10).cuda().double().eval()
        with torch.no_grad():net.head.weight.normal_(std=.01)
        x=torch.randn(2,768,10,device='cuda',dtype=torch.float64);y=torch.randn_like(x);q=torch.zeros(2,24,1,device='cuda',dtype=torch.float64)
        with torch.no_grad():
            error=(net(q,x+y)[...,5]-net(q,x)[...,5]-net(q,y)[...,5]+net(q,torch.zeros_like(x))[...,5]).abs().max().item()
            core_error=(net.core(x)-net.core.reference(x)).abs().max().item()
        assert error<1e-9 and core_error<1e-9
        checks[family]=dict(affine_superposition_error=error,recurrence_error=core_error,quarter_tick_hours=.25)
    t.write(t.ROOT/'linearity_smoke.json',dict(passed=True,checks=checks));print('LINEARITY SMOKE',checks,flush=True)


def main():
    t.setup(17);rows=[];checks=[]
    for folder in sorted((t.ROOT/'folds').iterdir()):
        d=np.load(folder/'data.npz');selection=json.loads((folder/'selection.json').read_text());nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc'])
        for family in t.FAMILIES:
            chosen=selection['by_recipe'][family];ck=torch.load(folder/'trials'/str(chosen['config']['id'])/'model.pt',map_location='cpu',weights_only=False)
            net=t.QuantileSSM(ck['config'],1,10).double().eval();net.load_state_dict(ck['state'])
            path=folder/'interpretations'/family;path.mkdir(parents=True,exist_ok=True)
            # First test day is deterministic, not chosen for its test outcome.
            index=nt+ns+nc;x=d['r'][index:index+1].astype('float64');q=d['q'][index:index+1].astype('float64')
            with torch.no_grad():
                xt=torch.tensor(x);qt=torch.tensor(q);reference=net(qt,xt)[...,5].numpy()[0]*20
                zero=net(qt,torch.zeros_like(xt))[...,5].numpy()[0]*20
                contributions=[]
                for channel in range(10):
                    masked=torch.zeros_like(xt);masked[:,:,channel]=xt[:,:,channel]
                    contributions.append(net(qt,masked)[...,5].numpy()[0]*20-zero)
                channel_contributions=np.asarray(contributions)
                add_error=float(np.abs(channel_contributions.sum(0)+zero-reference).max());assert add_error<1e-8
                kernel=net.core.kernel(768).numpy();direct=net.core.D.numpy();W=net.head.weight[:4].numpy()
                # Independent numpy convolution reproduces exact point corrections.
                full_kernel=kernel.copy();full_kernel[:,0]+=direct
                forward=np.stack([np.convolve(x[0,:,ch],full_kernel[ch])[:768] for ch in range(10)],-1)
                backward=np.stack([np.convolve(x[0,::-1,ch],full_kernel[ch])[:768][::-1] for ch in range(10)],-1)
                predicted=(np.concatenate([forward,backward],-1)[-96:]@W.T+net.head.bias[:4].numpy()).T.reshape(4,24,4)
                predicted=20*(predicted-predicted.mean(-1,keepdims=True))
                conv_error=float(np.abs(predicted-reference).max());assert conv_error<1e-8
                # Exact normalized-input derivative for hours0,12,23, including conservation.
                times=np.concatenate([672+4*h+np.arange(4) for h in [0,12,23]]);source=np.arange(768)
                delta=times[:,None]-source[None,:]
                fk=full_kernel[:,np.maximum(delta,0)]*(delta>=0)[None]
                bk=full_kernel[:,np.maximum(-delta,0)]*(delta<=0)[None]
                jac=W[:,:10,None,None]*fk[None]+W[:,10:,None,None]*bk[None]  # zone,channel,12outputs,768sources
                jac=jac.transpose(0,2,1,3).reshape(4,3,4,10,768);jac=20*(jac-jac.mean(2,keepdims=True))
                a,b=net.core.discrete();a=a.numpy();b=b.numpy();C=torch.view_as_complex(net.core.C.contiguous()).numpy()
                point_weights=np.abs(W[:,:10])+np.abs(W[:,10:]);strength=2*np.abs(C*b)*point_weights.sum(0)[:,None]
                for ch in range(10):
                    for mode in range(a.shape[1]):
                        angle=abs(float(np.angle(a[ch,mode])));radius=abs(a[ch,mode]);period=2*np.pi*.25/angle if angle>1e-10 else None
                        decay=-.25/np.log(radius) if 0<radius<1 else None
                        rows.append(dict(month=folder.name,family=family,channel=NAMES[ch],mode=mode,period_hours=period,decay_hours=decay,amplitude_weight=float(strength[ch,mode]),discrete_angle=float(np.angle(a[ch,mode]))))
            np.savez_compressed(path/'exact_explanation.npz',date=d['dates'][index],baseline=d['b'][index],point=d['b'][index]+reference,zero_input_correction=zero,channel_contributions=channel_contributions,input_names=NAMES,input_normalized=x[0],lag_kernel=kernel,feedthrough=direct,median_readout=W,effective_kernel_hours=np.array([0,12,23]),effective_kernel=jac,normalizer_mu=d['mur'],normalizer_sd=d['sdr'])
            checks.append(dict(month=folder.name,family=family,contribution_error=add_error,numpy_convolution_error=conv_error,total_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad)))
    frame=pd.DataFrame(rows);frame.to_csv(t.ROOT/'frequencies.csv',index=False);t.write(t.ROOT/'interpretability_audit.json',dict(passed=True,exact_linearity=True,models=checks))
    # Static research figure, no claims of uniquely identified physical mechanisms.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for family,offset in [('learned_frequencies',-.1),('fixed_frequencies',.1)]:
        sub=frame[(frame.family==family)&frame.period_hours.notna()];month=pd.Categorical(sub.month,categories=sorted(frame.month.unique())).codes
        weight=sub.amplitude_weight.to_numpy();size=4+35*weight/max(float(weight.max()),1e-12)
        ax.scatter(month+offset,sub.period_hours,s=size,alpha=.3,label=family)
    ax.set_yscale('log');ax.set_xticks(range(8),sorted(frame.month.unique()),rotation=30);ax.set_ylabel('Aliased discrete-mode period (hours)');ax.set_title('Single-layer S4D modes; marker size shows point-readout amplitude');ax.legend();fig.tight_layout();fig.savefig(t.ROOT/'frequencies.png',dpi=160);plt.close(fig)
    with (t.ROOT/'REPORT.md').open('a') as out:out.write('\nPoint path: one shared bidirectional linear S4D and a linear readout, followed by hourly mean projection. No nonlinear features, activation, LayerNorm or clipping. Quantile gaps use a separate softplus head. Exact source-channel contributions and independent convolution replay passed for all16 family/fold winners. Frequencies and decay times are learned filter modes, not uniquely identified physical cycles; inspect effective kernels and cross-fold consistency.\n')
    print('INTERPRETABILITY AUDIT PASSED',len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();smoke() if a.smoke else main()
