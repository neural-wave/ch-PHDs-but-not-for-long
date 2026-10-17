"""Parallel quarter/hour rows, joint zones, and a small late-information head."""
import datetime as dt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AUX = 23  # 4 current prices, 8 neighboring changes, 4 quarter flags, 7 weekday flags.

def build_inputs(dates, hourly, history_dates, history_y, history):
    hist={str(d):y for d,y in zip(history_dates,history_y)};out=[]
    for date,h in zip(dates,hourly):
        day=dt.date.fromisoformat(str(date));pairs=[];masks=[]
        for lag in (range(7,-1,-1) if history else [0]):
            key=str(day-dt.timedelta(days=lag))
            if lag:
                assert key<str(date)
                yy=hist.get(key);available=yy is not None
                yy=np.zeros((4,24,4)) if yy is None else yy
                hh=yy.mean(-1)
            else:
                # Query-day quarter prices are never read, even during training.
                yy=np.zeros((4,24,4));hh=h;available=False
            pairs.extend([yy.reshape(4,96)/100,np.repeat(hh,4,axis=-1)/100])
            masks.extend([np.full((1,96),float(available)),np.full((1,96),float(available or lag==0))])
        prices=np.stack(pairs,axis=1)  # zone, row, quarter; Q/H pairs for each day.
        mask=np.stack(masks,axis=1)
        grid=np.concatenate([prices,mask],axis=0).reshape(-1,96)
        left=h-np.pad(h,((0,0),(1,0)),mode='edge')[:,:24]
        right=np.pad(h,((0,0),(0,1)),mode='edge')[:,1:]-h
        aux=np.concatenate([np.repeat(h,4,axis=-1)/100,np.repeat(left,4,axis=-1)/100,
                            np.repeat(right,4,axis=-1)/100,np.eye(4)[np.arange(96)%4].T,
                            np.broadcast_to(np.eye(7)[day.weekday()][:,None],(7,96))])
        out.append(np.concatenate([grid,aux]))
    return np.asarray(out,dtype=np.float32)

class ParallelCNN(nn.Module):
    def __init__(self,n,cfg):
        super().__init__();self.rows=(n-AUX)//5;assert self.rows in [2,16]
        self.conv=nn.Conv2d(5,8,(2,8),stride=(2,1))
        self.fusion=nn.Conv1d(8*(self.rows//2)+AUX,8,1)
        self.output=nn.Conv1d(8,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,x):
        grid=x[:,:-AUX].reshape(-1,5,self.rows,96)
        # Even kernel: three quarters on the left, four on the right.
        return F.relu(self.conv(F.pad(grid,(3,4,0,0),mode='replicate')))
    def forward(self,x):
        encoded=self.encode(x).flatten(1,2)
        fused=F.relu(self.fusion(torch.cat([encoded,x[:,-AUX:]],dim=1)))
        return self.output(fused)

def smoke(d,configs,root,model):
    dates=d['dates'];h=d['h'];y=d['y'];hist=dict(zip(dates,y))
    for history in [False,True]:
        raw=build_inputs(dates[333:334],h[333:334],dates,y,history)
        rows=16 if history else 2;grid=raw[:,:-AUX].reshape(1,5,rows,96)
        assert not grid[0,:4,-2].any() and not grid[0,4,-2].any()
        np.testing.assert_allclose(grid[0,:4,-1],np.repeat(h[333],4,axis=-1)/100,rtol=1e-6)
        assert grid[0,4,-1].all()
        if history:
            for i,lag in enumerate(range(7,0,-1)):
                key=str(dt.date.fromisoformat(str(dates[333]))-dt.timedelta(days=lag))
                np.testing.assert_allclose(grid[0,:4,2*i],hist[key].reshape(4,96)/100,rtol=1e-6)
                np.testing.assert_allclose(grid[0,:4,2*i+1],np.repeat(hist[key].mean(-1),4,axis=-1)/100,rtol=1e-6)
            missing=build_inputs(dates[333:334],h[333:334],[],[],True)[:,:-AUX].reshape(1,5,16,96)
            assert not missing[:,:,:14].any()
        f=np.load(root/f'features_{int(history)}.npz');cfg=next(c for c in configs if c['history']==history)
        net=model(f['x'].shape[1],cfg).cuda()
        nn.init.normal_(net.net.output.weight,std=.01)
        xx=torch.tensor(f['x'][:2],device='cuda',requires_grad=True);pred=net(xx)
        pred[:,0,:,0].sum().backward()
        assert xx.grad[:,rows+rows-1].abs().sum()>0  # zone1, current hourly row.
        assert pred.mean(-1).abs().max()<1e-6
        assert net.net.conv.weight.shape==(8,5,2,8)

def export_filters(root,model):
    import json
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selection=json.loads((root/'selection.json').read_text())
    for key,winner in selection.items():
        folder=root/'trials'/str(winner['config']['id']);checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        f=np.load(root/f'features_{key}.npz');net=model(f['x'].shape[1],winner['config']).cuda().eval();net.load_state_dict(checkpoint['state'])
        weights=net.net.conv.weight.detach().cpu().numpy()
        np.savez(folder/'learned_filters.npz',conv2d=weights,fusion=net.net.fusion.weight.detach().cpu().numpy(),
                 output=net.net.output.weight.detach().cpu().numpy(),mu=f['mu'],sd=f['sd'])
        for channel in range(5):
            fig,axes=plt.subplots(4,2,figsize=(9,7),layout='constrained');lim=max(float(np.abs(weights[:,channel]).max()),1e-8)
            for i,ax in enumerate(axes.flat):
                im=ax.imshow(weights[i,channel],aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
                ax.set(title=f'Filter {i}',yticks=[0,1],yticklabels=['Quarter prices','Hourly repeated'],xticks=range(8),xticklabels=range(-3,5))
            fig.suptitle(f"{'7 days' if key=='1' else 'No history'}: {'ZONE_'+str(channel) if channel<4 else 'Availability mask'}; normalized-input weights")
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6)
            fig.savefig(folder/f'kernels_channel_{channel}.png',dpi=140);plt.close(fig)
        with torch.no_grad():activation=net.net.encode(torch.tensor(f['x'][302:303],device='cuda')).cpu().numpy()[0]
        fig,axes=plt.subplots(4,2,figsize=(12,7),layout='constrained')
        for i,ax in enumerate(axes.flat):
            ax.imshow(activation[i],aspect='auto',origin='lower',cmap='viridis')
            ax.set(title=f'Filter {i}',xlabel='Quarter of day',ylabel='Day (-7..0)' if key=='1' else 'Current day')
            if key=='1':ax.set_yticks(range(8),range(-7,1))
        fig.suptitle('Activations on 2026-08-01; interpretation example, not causal attribution')
        fig.savefig(folder/'activations_august01.png',dpi=140);plt.close(fig)
