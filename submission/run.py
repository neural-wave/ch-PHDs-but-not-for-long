"""Offline inference, with no fitting, calibration or external lookups."""
import argparse
from pathlib import Path
from predict import predict
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--hourly',required=True);p.add_argument('--history',required=True);p.add_argument('--output',required=True);p.add_argument('--model',default=str(Path(__file__).with_name('model.pkl')));a=p.parse_args()
 predict(a.model,a.hourly,a.history,a.output)
