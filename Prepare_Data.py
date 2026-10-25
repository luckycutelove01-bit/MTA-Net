
import json
from functools import partial
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import BatchSampler,DataLoader
from Datasets.read_labels import read_records
from Datasets.data_loader import read_image_batch

class SafeBatchSampler(BatchSampler):
    """Same seed+epoch shuffle and singleton-tail dropping as revised source."""
    def __init__(self,indices: Iterable[int],batch_size=8,seed=21,shuffle=True):
        self.indices=list(indices);self.batch_size=int(batch_size)
        if self.batch_size<2:raise ValueError('BatchNorm training requires batch_size >= 2')
        self.seed=int(seed);self.shuffle=bool(shuffle);self.epoch=0
    def __iter__(self):
        indices=list(self.indices)
        if self.shuffle:
            order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            indices=[indices[i] for i in order]
        self.epoch+=1
        batches=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        if batches and len(batches[-1])==1:batches.pop()
        yield from batches
    def __len__(self):
        n=len(self.indices)
        return (n+self.batch_size-1)//self.batch_size-int(n%self.batch_size==1)

def read_split(path):
    manifest=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if manifest.get('split_seed')!=21 or manifest.get('sample_swaps') is not False:
        raise ValueError('Expected fixed seed-21 manifest without sample exchanges')
    groups={k:manifest[k] for k in ('train','val','test')}
    flat=sum(groups.values(),[])
    if [len(groups[k]) for k in groups]!=[145,26,31] or len(set(flat))!=202:
        raise ValueError('Expected 202 distinct IDs with counts 145/26/31')
    return groups

def prepare_loaders(config,test_only=False):
    records=read_records(config.image_dir,config.labels)
    groups=read_split(config.split);lookup={r.sample_id:r for r in records}
    needed=groups['test'] if test_only else sum(groups.values(),[])
    missing=sorted(set(needed)-set(lookup))
    if missing:raise ValueError(f'Missing {len(missing)} required samples: {missing}; the 101 bundled examples are not the complete experiment.')
    if not test_only and set(lookup)!=set(needed):raise ValueError('Full training requires exactly the archived 202 samples')
    loaders={}
    for subset in (('test',) if test_only else ('train','val','test')):
        dataset=[lookup[sid] for sid in groups[subset]]
        collate=partial(read_image_batch,training=subset=='train')
        if subset=='train':
            sampler=SafeBatchSampler(range(len(dataset)),config.batch_size,config.seed)
            loaders[subset]=DataLoader(dataset,batch_sampler=sampler,num_workers=config.num_workers,collate_fn=collate)
        else:
            loaders[subset]=DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,collate_fn=collate)
    return loaders,groups
