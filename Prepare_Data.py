
import json
from functools import partial
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import BatchSampler,DataLoader
from Datasets.read_labels import read_records
from Datasets.data_loader import read_image_batch

class SafeBatchSampler(BatchSampler):
    """Shuffle by seed and epoch; rebalance singleton tails for BatchNorm."""
    def __init__(self,indices: Iterable[int],batch_size=8,seed=21,shuffle=True):
        self.indices=list(indices);self.batch_size=int(batch_size)
        if self.batch_size<2:raise ValueError('BatchNorm training requires batch_size >= 2')
        if len(self.indices)==1:raise ValueError('BatchNorm training requires at least two samples')
        self.seed=int(seed);self.shuffle=bool(shuffle);self.epoch=0
    def __iter__(self):
        indices=list(self.indices)
        if self.shuffle:
            order=torch.randperm(len(indices),generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            indices=[indices[i] for i in order]
        self.epoch+=1
        batches=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        if len(batches)>1 and len(batches[-1])==1:
            if len(batches[-2])>2:
                batches[-1].insert(0,batches[-2].pop())
            else:
                batches[-2].extend(batches.pop())
        yield from batches
    def __len__(self):
        n=len(self.indices)
        return (n+self.batch_size-1)//self.batch_size-int(self.batch_size==2 and n>2 and n%2==1)

def read_split(path):
    manifest=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    groups={k:manifest[k] for k in ('train','val','test')}
    flat=sum(groups.values(),[])
    counts=[len(groups[k]) for k in groups]
    if counts not in ([145,26,31],[143,20,39]) or len(flat)!=202 or len(set(flat))!=202:
        raise ValueError('Expected 202 distinct IDs with counts 145/26/31 or 143/20/39')
    return groups

def prepare_loaders(config,test_only=False,texture_extractor=None):
    records=read_records(config.image_dir,config.labels)
    groups=read_split(config.split);lookup={r.sample_id:r for r in records}
    needed=groups['test'] if test_only else sum(groups.values(),[])
    missing=sorted(set(needed)-set(lookup))
    if missing:raise ValueError(f'Required sample IDs have no image-label record: {missing}')
    if not test_only and set(lookup)!=set(needed):raise ValueError('Training labels must match the 202 IDs in the split file')
    loaders={}
    for subset in (('test',) if test_only else ('train','val','test')):
        dataset=[lookup[sid] for sid in groups[subset]]
        collate=partial(read_image_batch,training=subset=='train',texture_extractor=texture_extractor)
        if subset=='train':
            sampler=SafeBatchSampler(range(len(dataset)),config.batch_size,config.seed)
            loaders[subset]=DataLoader(dataset,batch_sampler=sampler,num_workers=config.num_workers,collate_fn=collate)
        else:
            loaders[subset]=DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,collate_fn=collate)
    return loaders,groups
