"""Train MTA-Net for one model seed on a fixed data split."""
import argparse
from Parameters import RunParameters
from Utils.Network_functions import train_model,set_seed

def main():
    defaults=RunParameters()
    parser=argparse.ArgumentParser(description='Train MTA-Net for one model seed on a fixed data split.')
    for name in ('image_dir','labels','split','output','device'):
        parser.add_argument('--'+name.replace('_','-'),default=getattr(defaults,name))
    parser.add_argument('--seed',type=int,default=defaults.seed)
    parser.add_argument('--epochs',type=int,default=defaults.epochs)
    parser.add_argument('--batch-size',type=int,default=defaults.batch_size)
    args=parser.parse_args()
    from Prepare_Data import prepare_loaders
    config=RunParameters(**vars(args));set_seed(config.seed)
    loaders,split=prepare_loaders(config)
    metrics=train_model(config,loaders,split)
    print({k:v for k,v in metrics.items() if k!='per_bin'})

if __name__=='__main__':main()
