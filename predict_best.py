"""Offline entry point using the frozen validation-selected manifest."""
import argparse,json
from pathlib import Path
from predict import predict
from select_best import NAMES

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',default='results/monthly_best.json')
    p.add_argument('--hourly',required=True);p.add_argument('--history',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();selection=Path(a.selection);meta=json.loads(selection.read_text())
    assert meta['selected_by'] in ['validation_mae_only','forward_28day_cv_mae','august_validation_mae','frozen_august_pipeline_refit_all_official_days']
    assert meta['best_directory'] in NAMES+['rolling_search_gpu1','monthly_base_gpu1','monthly_cat_gpu1','monthly_graph_gpu1','monthly_revin_gpu1','monthly_joint_blend_gpu1','deployment_all363_gpu1']
    model=selection.parent/meta['best_directory']/'model.pkl'
    predict(model,a.hourly,a.history,a.output)
