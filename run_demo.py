"""Run from any directory: python /path/to/run_demo.py [--paper]."""
from pathlib import Path
import sys, json, argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from handoff import ControlGMM1,GMM2,PaperGMM1,history_features

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paper',action='store_true');args=parser.parse_args()
    if args.paper:
        d=json.loads((ROOT/'examples/paper_k15.json').read_text(encoding='utf-8'))
        print(json.dumps({'paper_K15_scalar_residual':PaperGMM1().predict(d['x']).tolist()}));return
    g1=ControlGMM1();g2c=GMM2('current');g2h=GMM2('history')
    for sid in json.loads((ROOT/'examples/index.json').read_text()):
        folder=ROOT/'examples'/sid;d=json.loads((folder/'input.json').read_text())
        features={**d['current'],**history_features(pd.read_csv(folder/'history.csv'))}
        p=g1.predict(**d['gmm1'])
        print(json.dumps({'scenario':sid,'gmm1_first_delta_ay_r':p.mu_deltaY[0].tolist(),
                          'gmm2_current_q':g2c.predict(features)['mean'][0].tolist(),
                          'gmm2_history_q':g2h.predict(features)['mean'][0].tolist()},ensure_ascii=False))
if __name__=='__main__':main()
