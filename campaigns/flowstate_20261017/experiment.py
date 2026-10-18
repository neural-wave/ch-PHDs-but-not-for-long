"""Exploratory pretrained FlowState + current-hourly-curve reconstruction.

Excluded from submission. Run on the existing Vast GPU, never on the Mac.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE / 'granite-tsfm'))
import numpy as np
import torch
from torch import nn
from reconstruction import read_targets, baseline

ROOT = PROJECT / 'results/flowstate_20261017'
SOURCE_SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
CODE_SHA = '640f7c80a21a4e2253b6d6140d067452bb93e7d4'
START, TRAIN_END, VAL_END = 28, 302, 333
CHANNELS = 94  # 23 query/calendar, 7*(4 Q + 4 H + 1 mask), 8 FlowState.


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_data():
    source = PROJECT / 'train_prices_15min.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    dates, y = read_targets(source)
    assert len(dates) == 363 and dates[302] == '2026-08-01' and dates[333] == '2026-09-01'
    return dates.astype('U10'), y, y.mean(-1)


def history_context(date, dates, y):
    """28 calendar days: preserve skipped DST days as 96 missing positions."""
    hist = dict(zip(dates, y))
    day = dt.date.fromisoformat(str(date))
    rows = []
    for lag in range(28, 0, -1):
        key = str(day - dt.timedelta(days=lag))
        assert key < str(date)
        rows.append(hist[key].reshape(4, 96) if key in hist else np.full((4, 96), np.nan))
    return np.concatenate(rows, axis=1).astype('float32')


def inputs(date, hourly, dates, y, forecast):
    """No quarter-hour query target is read; only strictly earlier history."""
    hist = dict(zip(dates, y))
    day = dt.date.fromisoformat(str(date))
    left = hourly - np.pad(hourly, ((0, 0), (1, 0)), mode='edge')[:, :24]
    right = np.pad(hourly, ((0, 0), (0, 1)), mode='edge')[:, 1:] - hourly
    rows = [np.repeat(hourly, 4, axis=1) / 100,
            np.repeat(left, 4, axis=1) / 100, np.repeat(right, 4, axis=1) / 100,
            np.eye(4)[np.arange(96) % 4].T,
            np.broadcast_to(np.eye(7)[day.weekday()][:, None], (7, 96))]
    for lag in range(7, 0, -1):
        key = str(day - dt.timedelta(days=lag))
        assert key < str(date)
        prior = hist.get(key)
        available = prior is not None
        prior = np.zeros((4, 24, 4)) if prior is None else prior
        rows.extend([prior.reshape(4, 96) / 100,
                     np.repeat(prior.mean(-1), 4, axis=1) / 100,
                     np.full((1, 96), float(available))])
    shape = forecast.reshape(4, 24, 4)
    shape = shape - shape.mean(-1, keepdims=True)
    rows.extend([forecast / 100, shape.reshape(4, 96) / 20])
    result = np.concatenate(rows).astype('float32')
    assert result.shape == (CHANNELS, 96) and np.isfinite(result).all()
    return result


def normalize(raw, mu, sd, flowstate):
    x = np.clip((raw - mu) / sd, -15, 15).astype('float32')
    for i in range(7):
        mask = raw[:, 23 + i * 9 + 8:23 + i * 9 + 9]
        x[:, 23 + i * 9:23 + i * 9 + 8] *= mask
        x[:, 23 + i * 9 + 8:23 + i * 9 + 9] = mask
    if not flowstate:
        x[:, -8:] = 0  # Same parameter count; identical current-day/history inputs.
    return x


class ConditionalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv1d(CHANNELS, 16, 5, padding=2, padding_mode='replicate')
        self.full_curve = nn.Linear(4 * 24, 16)
        self.output = nn.Conv1d(16, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # All 24 hourly prices of all 4 zones condition every output position.
        curve = self.full_curve(x[:, :4, ::4].reshape(-1, 96))[:, :, None]
        out = self.output(torch.relu(self.local(x) + curve)).reshape(-1, 4, 24, 4)
        return out - out.mean(-1, keepdim=True)


def flowstate_features(start, end):
    from huggingface_hub import HfApi
    from tsfm_public import FlowStateForPrediction
    dates, y, h = load_data()
    code = subprocess.check_output(['git', '-C', str(HERE / 'granite-tsfm'), 'rev-parse', 'HEAD'], text=True).strip()
    assert code == CODE_SHA
    receipt_path = ROOT / 'model_receipt.json'
    revision = (json.loads(receipt_path.read_text())['checkpoint_sha'] if receipt_path.exists()
                else HfApi().model_info('ibm-research/flowstate', revision='r1.1').sha)
    model, info = FlowStateForPrediction.from_pretrained('ibm-research/flowstate', revision=revision,
                                                        output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
    model = model.cuda().eval().requires_grad_(False)
    configure()
    write(receipt_path, dict(repository='ibm-research/flowstate', requested_revision='r1.1',
          checkpoint_sha=revision, code_sha=code, loading_info=info,
          parameters=sum(p.numel() for p in model.parameters()),
          frozen=True, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
          scale_factor=.25, horizon=96, context_calendar_days=28,
          point='native median', note='Pretrained external weights: exploratory, ineligible for submission'))
    for a in range(start, end, 4):
        b = min(a + 4, end)
        contexts = np.concatenate([history_context(d, dates, y) for d in dates[a:b]])
        with torch.inference_mode():
            f = model(torch.as_tensor(contexts[..., None], device='cuda'), batch_first=True,
                      scale_factor=.25, prediction_length=96, prediction_type='median').prediction_outputs
        assert tuple(f.shape) == ((b - a) * 4, 96, 1)
        f = f.cpu().numpy().reshape(b - a, 4, 96).astype('float64')
        assert np.isfinite(f).all()
        for j, i in enumerate(range(a, b)):
            np.savez(ROOT / 'forecasts' / f'{dates[i]}.npz', forecast=f[j])
        write(ROOT / 'progress.json', dict(stage='frozen FlowState inference', completed=b-start,
              total=end-start, last_date=str(dates[b-1])))
        print('FLOWSTATE', str(dates[a]), str(dates[b-1]), flush=True)
    del model
    torch.cuda.empty_cache()


def raw_features(start, end):
    dates, y, h = load_data()
    return np.stack([inputs(dates[i], h[i], dates, y,
              np.load(ROOT / 'forecasts' / f'{dates[i]}.npz')['forecast']) for i in range(start, end)])


def fit(flowstate, seed, raw, epochs=300, smoke=False):
    configure()
    torch.manual_seed(seed)
    dates, y, h = load_data()
    mu = raw[:TRAIN_END-START].mean((0, 2), keepdims=True)
    sd = raw[:TRAIN_END-START].std((0, 2), keepdims=True).clip(.05)
    x = torch.as_tensor(normalize(raw, mu, sd, flowstate), device='cuda')
    target = torch.as_tensor((y[START:VAL_END] - baseline(h[START:VAL_END])) / 20,
                             dtype=torch.float32, device='cuda')
    net = ConditionalCNN().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    n = TRAIN_END - START
    best, best_epoch, state = float('inf'), 0, None
    curves = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        total = 0.
        for ids in torch.randperm(n, device='cuda').split(32):
            loss = (net(x[ids]) - target[ids]).abs().mean()
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(ids)
        net.eval()
        with torch.no_grad():
            val = float((net(x[n:]) - target[n:]).abs().mean()) * 20
        curves.append([epoch, total / n * 20, val])
        if val < best:
            best, best_epoch = val, epoch
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch - best_epoch >= 30:
            break
    folder = ROOT / ('smoke_fits' if smoke else 'trials') / f'{int(flowstate)}_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=state, mu=mu, sd=sd, flowstate=flowstate, seed=seed), folder / 'model.pt')
    np.savetxt(folder / 'curves.csv', curves, delimiter=',', header='epoch,train_mae,august_mae', comments='')
    result = dict(flowstate=flowstate, seed=seed, validation_mae=best, best_epoch=best_epoch,
                  epochs=epoch, seconds=time.time()-started,
                  parameters=sum(p.numel() for p in net.parameters()), folder=str(folder.relative_to(ROOT)))
    write(folder / 'fit.json', result)
    print('FIT', result, flush=True)
    return result


def predict(row, raw):
    ck = torch.load(ROOT / row['folder'] / 'model.pt', weights_only=False, map_location='cpu')
    net = ConditionalCNN().cuda().eval(); net.load_state_dict(ck['state'])
    x = normalize(raw, ck['mu'], ck['sd'], ck['flowstate'])
    with torch.inference_mode():
        residual = net(torch.as_tensor(x, device='cuda')).cpu().numpy().astype('float64')
    return residual - residual.mean(-1, keepdims=True)


def smoke():
    configure()
    dates, y, h = load_data()
    date = dates[TRAIN_END]
    forecast = np.load(ROOT / 'forecasts' / f'{date}.npz')['forecast']
    a = inputs(date, h[TRAIN_END], dates, y, forecast)
    changed = y.copy(); changed[TRAIN_END:] += np.arange(4) * 77777
    np.testing.assert_array_equal(a, inputs(date, h[TRAIN_END], dates, changed, forecast))
    np.testing.assert_array_equal(history_context(date, dates, y), history_context(date, dates, changed))
    np.testing.assert_array_equal(a, inputs(date, h[TRAIN_END], dates[:TRAIN_END], y[:TRAIN_END], forecast))
    raw = raw_features(START, VAL_END)
    mu = raw[:TRAIN_END-START].mean((0, 2), keepdims=True)
    sd = raw[:TRAIN_END-START].std((0, 2), keepdims=True).clip(.05)
    torch.manual_seed(17)
    net = ConditionalCNN().cuda()
    nn.init.normal_(net.output.weight, std=.01)
    x = torch.tensor(normalize(raw[-1:], mu, sd, True), device='cuda', requires_grad=True)
    point = net(x); point[0, 0, 0, 0].backward()
    assert x.grad[0, 1, 92].abs() > 0  # Zone1 hour24 changes Zone0 hour1 shape.
    assert x.grad[0, -8:].abs().sum() > 0
    assert float(point.mean(-1).abs().max().detach()) < 1e-6
    fits = [fit(arm, 17, raw, epochs=2, smoke=True) for arm in [False, True]]
    for row in fits:
        assert np.isfinite(predict(row, raw[-2:])).all()
    write(ROOT / 'smoke.json', dict(passed=True, no_query_target_leakage=True,
          missing_calendar_dates_preserved=True, full_curve_cross_zone_gradient=True,
          flowstate_gradient=True, frozen_checkpoint_receipt=True, two_epoch_fits=fits))


def train():
    raw = raw_features(START, VAL_END)
    rows = [fit(arm, seed, raw) for arm in [False, True] for seed in [17, 43, 71]]
    selected = {str(int(arm)): min((r for r in rows if r['flowstate'] == arm),
                                  key=lambda r: r['validation_mae']) for arm in [False, True]}
    write(ROOT / 'selection.json', dict(by_arm=selected, candidates=rows,
          selection='August only; persisted before September FlowState inference or scoring',
          frozen_at_unix=time.time()))


def score():
    configure()
    dates, y, h = load_data()
    raw = raw_features(TRAIN_END, len(dates))
    selection = json.loads((ROOT / 'selection.json').read_text())
    predictions = {}
    for key, row in selection['by_arm'].items():
        predictions['conditional_' + ('flowstate' if key == '1' else 'control')] = (
            baseline(h[TRAIN_END:]) + 20 * predict(row, raw))
    forecast = np.stack([np.load(ROOT / 'forecasts' / f'{d}.npz')['forecast'] for d in dates[TRAIN_END:]])
    forecast = forecast.reshape(-1, 4, 24, 4)
    predictions['flowstate_projected_only'] = h[TRAIN_END:, ..., None] + forecast - forecast.mean(-1, keepdims=True)
    predictions['interpolation'] = baseline(h[TRAIN_END:])
    predictions['hourly_mean'] = np.repeat(h[TRAIN_END:, ..., None], 4, axis=-1)
    metrics = {}
    for name, pred in predictions.items():
        err = np.abs(pred[VAL_END-TRAIN_END:] - y[VAL_END:])
        mean_error = float(np.abs(pred.mean(-1) - h[TRAIN_END:]).max())
        assert np.isfinite(pred).all() and mean_error < 1e-8
        metrics[name] = dict(september_mae=float(err.mean()), by_zone=err.mean((0, 2, 3)).tolist(),
              daily_mae=err.mean((1, 2, 3)).tolist(), max_hourly_mean_error=mean_error,
              august_mae=float(np.abs(pred[:VAL_END-TRAIN_END] - y[TRAIN_END:VAL_END]).mean()))
    np.savez(ROOT / 'predictions.npz', dates=dates[TRAIN_END:], **predictions)
    write(ROOT / 'metrics.json', dict(methods=metrics, days=30, zones=4,
          caveat='September developmental, not blind. External pretrained weights; ineligible for submission.',
          flowstate_minus_control=metrics['conditional_flowstate']['september_mae']-metrics['conditional_control']['september_mae']))
    print('METRICS', metrics, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['features', 'smoke', 'train', 'score'])
    p.add_argument('--start', type=int, default=START)
    p.add_argument('--end', type=int, default=VAL_END)
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT / 'forecasts').mkdir(exist_ok=True)
    if args.stage == 'features': flowstate_features(args.start, args.end)
    else: {'smoke': smoke, 'train': train, 'score': score}[args.stage]()
