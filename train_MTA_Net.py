"""Single-seed main-model training. Nothing runs on import."""
import argparse,sys
from Parameters import RunParameters
from pathlib import Path
from Utils.Network_functions import train_model,set_seed

def main():
    defaults=RunParameters()
    parser=argparse.ArgumentParser(description='Train revised MTA-Net for ONE seed on the archived split.')
    for name in ('image_dir','labels','split','output','device'):
        parser.add_argument('--'+name.replace('_','-'),default=getattr(defaults,name))
    parser.add_argument('--seed',type=int,default=defaults.seed)
    parser.add_argument('--epochs',type=int,default=defaults.epochs)
    parser.add_argument('--batch-size',type=int,default=defaults.batch_size)
    args=parser.parse_args()
    # Data-loading implementation intentionally excluded from this delivery.
    if not Path(__file__).with_name('Prepare_Data.py').is_file():
        raise SystemExit('Data loading code was intentionally removed. Implement a local Prepare_Data.py with prepare_loaders(config, test_only=False) before running training or testing.')
    from Prepare_Data import prepare_loaders
    if sys.version_info[:2]!=(3,11):
        raise RuntimeError('Formal manuscript training requires Python 3.11. Import/forward checks can use other compatible environments.')
    config=RunParameters(**vars(args));set_seed(config.seed)
    loaders,split=prepare_loaders(config)
    metrics=train_model(config,loaders,split)
    print({k:v for k,v in metrics.items() if k!='per_bin'})

if __name__=='__main__':main()
