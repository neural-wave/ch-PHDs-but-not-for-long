"""Freeze a pipeline choice using completed validation scores only."""
import argparse,json
from pathlib import Path

NAMES=['search_gpu1','rich_search_gpu1','zone_search_gpu1','catboost_search_gpu1','projection_search_gpu1']

def select(results):
    root=Path(results);scores={}
    for name in NAMES:
        path=root/name/'metrics.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        if not report.get('offline_inference_qa','').startswith('passed'):continue
        scores[name]=report['validation']['mae']
    assert scores,'No completed and verified candidates'
    best=min(scores,key=scores.get)
    assert (root/best/'model.pkl').exists(),best+' is the winner but its model artifact is missing'
    result=dict(selected_by='validation_mae_only',best_directory=best,validation_scores=scores)
    temp=root/'best_search.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(root/'best_search.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args();select(a.results)
