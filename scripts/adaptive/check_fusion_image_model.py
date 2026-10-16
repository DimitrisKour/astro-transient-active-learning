#!/usr/bin/env python3
"""Architecture-only first-order check, no historical data or labels consumed."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, sha256, write_json_atomic
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model, load_checkpoint
from al_pipeline.astronomy.fusion_image.data import prepare_canonical_image


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--upstream-repo', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant', choices=('patch_images','btsbot_images'), default='patch_images')
    ap.add_argument('--image-initialization', choices=('pretrained','randomized'),default='randomized')
    ap.add_argument('--hf-snapshot', type=Path)
    ap.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    ap.add_argument('--seed', type=int, default=135)
    args=ap.parse_args()
    sha=subprocess.check_output(['git','-C',str(args.upstream_repo),'rev-parse','HEAD'],text=True).strip()
    spec=NewImageModelSpec(variant=args.variant,image_initialization=args.image_initialization,
         hf_snapshot=str(args.hf_snapshot) if args.hf_snapshot else None,
         hf_weights_sha256=sha256(args.hf_snapshot/'pytorch_model.bin') if args.hf_snapshot else None,source_revision=sha)
    device=torch.device(args.device)
    model, config=build_model(spec,seed=args.seed,device=device)
    model.train()
    # Batch >= 2 for coordinate-tower BatchNorm1d.
    metadata=torch.zeros((2,55),device=device,dtype=torch.float32)
    # The historical stored cutouts are 63x63. The upstream patch-image model
    # MUST receive its 49x49 central crop; BTSbot consumes 63x63. Apply the
    # same preprocessing routine used for real historical NPZ rows.
    canonical_images=torch.rand((2,3,63,63),dtype=torch.float32)
    image=torch.stack([
        prepare_canonical_image(img, runtime_config=config, augment=False)
        for img in canonical_images
    ]).to(device)
    target=torch.tensor([0,1],device=device)
    out=model(metadata,image)
    if not isinstance(out,dict) or 'logits' not in out:
        raise RuntimeError('Expected a logits dictionary')
    logits=out['logits']
    if tuple(logits.shape)!=(2,5) or not torch.isfinite(logits).all():
        raise RuntimeError(f'Invalid model logits: {tuple(logits.shape)}')
    loss=torch.nn.functional.cross_entropy(logits,target)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('No finite parameter gradients')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    cpath=args.out_dir/'architecture_roundtrip.pt'
    torch.save({'model_state':model.state_dict(), 'model_spec':asdict(spec),
                'class_names':list(CLASS_NAMES)},cpath)
    restored,_,_=load_checkpoint(cpath,spec=spec,device=device,seed=args.seed)
    restored.eval();model.eval()
    with torch.no_grad():
        a=model(metadata,image)['logits']
        b=restored(metadata,image)['logits']
    if not torch.allclose(a,b,atol=1e-6,rtol=1e-6):
        raise RuntimeError('Checkpoint round-trip changed logits')
    payload={'result':'architecture_forward_backward_reload_passed','upstream_sha':sha,
             'model_spec':asdict(spec),'input_metadata_features':55,
             'canonical_image_shape':[3,63,63],
             'model_input_image_shape':list(image.shape[1:]),
             'classes':list(CLASS_NAMES),
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'total_parameters':sum(p.numel() for p in model.parameters())}
    write_json_atomic(args.out_dir/'architecture_check.json',payload)
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
