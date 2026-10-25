"""Evaluate a specified revised checkpoint on the labelled fixed test set."""
import argparse
from Parameters import RunParameters
from pathlib import Path
from Utils.Network_functions import test_model as evaluate_checkpoint

def main():
    defaults=RunParameters()
    parser=argparse.ArgumentParser(description='Labelled 31-sample test evaluation; no standalone unlabelled-image predictor.')
    parser.add_argument('--checkpoint',required=True)
    for name in ('image_dir','labels','split','device'):
        parser.add_argument('--'+name.replace('_','-'),default=getattr(defaults,name))
    parser.add_argument('--output',default=str(__import__('pathlib').Path(defaults.output).parent/'test_seed21'))
    parser.add_argument('--batch-size',type=int,default=defaults.batch_size)
    args=parser.parse_args();checkpoint=args.checkpoint;del args.checkpoint
    # Data-loading implementation intentionally excluded from this delivery.
    if not Path(__file__).with_name('Prepare_Data.py').is_file():
        raise SystemExit('Data loading code was intentionally removed. Implement a local Prepare_Data.py with prepare_loaders(config, test_only=False) before running training or testing.')
    from Prepare_Data import prepare_loaders
    config=RunParameters(**vars(args))
    loaders,split=prepare_loaders(config,test_only=True)
    metrics=evaluate_checkpoint(checkpoint,config,loaders['test'],split)
    print({k:v for k,v in metrics.items() if k!='per_bin'})

if __name__=='__main__':main()
