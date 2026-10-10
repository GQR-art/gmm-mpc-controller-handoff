"""Optional reproduction using prepared features. Never runs during handoff tests."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='New empty directory for independently reproduced models')
    args=parser.parse_args();out=Path(args.output_dir).resolve()
    if out==ROOT or out.is_relative_to(ROOT/'models') or out.is_relative_to(ROOT/'src') or out.is_relative_to(ROOT/'data'):
        parser.error('Output must not overwrite frozen inputs, source or models')
    if out.exists() and any(out.iterdir()):parser.error('Output directory must be empty')
    import pandas as pd
    from future_work.short_horizon_output_residual.gmm2_200_lowdim_dynamic import pipeline
    pipeline.OUT=out;pipeline.MODELS=out/'models';pipeline.PLOTS=out/'plots'
    for p in [out,pipeline.MODELS,pipeline.PLOTS]:p.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/dynamic_transition_dataset.csv.gz')
    assert set(data.loc[data.used_for_gmm_fit,'split'])=={'train'}
    selected,scale=pipeline.train_models(data)
    # Evaluates only after validation-selected models have been saved.
    metrics,scenes,pred=pipeline.evaluate_partition(data[data.split=='test'],'test',scale)
    metrics.to_csv(out/'test_metrics.csv',index=False)
    scenes.to_csv(out/'test_per_scene.csv',index=False)
    print('Reproduction complete. Frozen distributed models were not changed.')
if __name__=='__main__':main()
