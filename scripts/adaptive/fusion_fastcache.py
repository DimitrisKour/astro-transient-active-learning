#!/usr/bin/env python3
"""CPU-only consolidation/verification and IO benchmark of full-history cache."""
import argparse
import json
import time
from pathlib import Path

from al_pipeline.astronomy.fusion_image.fast_cache import build, validate, FastHistoricalDataset
from al_pipeline.astronomy.fusion_image.contracts import load_rows, sha256

SOURCE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1')
TARGET = Path('/work/nvme/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_v1')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('build', 'check', 'bench'))
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--target', type=Path, default=TARGET)
    p.add_argument('--bench-rows', type=int, default=1024)
    p.add_argument('--scan-all', action='store_true', help='Scan full train and validation once through the DataLoader')
    args = p.parse_args()
    if args.command == 'build':
        print(json.dumps(build(source=args.source, target=args.target), indent=2), flush=True)
    elif args.command == 'check':
        print(json.dumps(validate(source=args.source, target=args.target), indent=2), flush=True)
    else:
        import numpy as np
        import torch
        from al_pipeline.astronomy.fusion_image.fast_cache import FastHistoricalDataset
        sourceplan = json.loads((args.source / 'plan.json').read_text())
        results = {'nvidia_gpu_used': False, 'io_cases': []}
        for split in ('train', 'validation'):
            rows = load_rows(sourceplan['manifests'][split])
            t = time.perf_counter()
            ds = FastHistoricalDataset(rows=rows, cache_root=args.target, split=split,
                                       object_column=sourceplan['id_column'],
                                       label_column=sourceplan['label_column'])
            startup = time.perf_counter() - t
            chosen = np.random.default_rng(135).choice(len(ds), size=min(args.bench_rows,len(ds)), replace=False)
            t = time.perf_counter()
            nbytes = 0
            for i in chosen:
                item = ds[int(i)]
                nbytes += item['image'].numel() * item['image'].element_size()
                nbytes += item['metadata'].numel() * item['metadata'].element_size()
            elapsed = time.perf_counter()-t
            entry={'split':split,'rows':len(ds),
                 'dataset_init_seconds':startup,'sample_count':len(chosen),
                 'sample_seconds':elapsed,'samples_per_second':len(chosen)/elapsed,
                 'bytes_sampled':nbytes}
            if args.scan_all:
                from torch.utils.data import DataLoader
                t_full=time.perf_counter()
                seen=0
                for batch in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
                    seen += len(batch['obj_id'])
                entry['full_scan_rows']=seen
                entry['full_scan_seconds']=time.perf_counter()-t_full
                entry['full_scan_rows_per_second']=seen/entry['full_scan_seconds']
                if seen!=len(ds):raise RuntimeError('Partial full-split scan')
            results['io_cases'].append(entry)
        print(json.dumps(results,indent=2),flush=True)

if __name__ == '__main__':
    main()
