#!/usr/bin/env python3
"""Fresh image+metadata model training check. Never submits jobs or uses test labels."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.backends import NewImageTrainer, ImageTrainingPolicy
from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, load_index, sha256, write_json_atomic
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.training.base import TrainingRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--new-index', type=Path, required=True)
    ap.add_argument('--new-root', type=Path)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--train-manifest', type=Path, required=True)
    ap.add_argument('--validation-manifest', type=Path, required=True)
    ap.add_argument('--upstream-repo', type=Path, required=True)
    ap.add_argument('--variant', choices=['patch_images', 'btsbot_images'], default='patch_images')
    ap.add_argument('--image-initialization', choices=['randomized', 'pretrained'], default='randomized')
    ap.add_argument('--hf-snapshot', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=135)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-batches', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()
    source_sha = subprocess.check_output(
        ['git', '-C', str(args.upstream_repo), 'rev-parse', 'HEAD'], text=True
    ).strip()
    reg = DatasetRegistry.from_toml(args.registry)
    dataset = reg.resolve_profile('astrominn_ztf').dataset
    spec = NewImageModelSpec(variant=args.variant, image_initialization=args.image_initialization,
                             hf_snapshot=str(args.hf_snapshot) if args.hf_snapshot else None,
         hf_weights_sha256=sha256(args.hf_snapshot/'pytorch_model.bin') if args.hf_snapshot else None,
                             source_revision=source_sha)
    index = load_index(args.new_index, root=args.new_root)
    policy = ImageTrainingPolicy(max_epochs=args.epochs, patience=max(1, args.epochs),
                                 batch_size=4, learning_rate=args.lr,
                                 max_training_batches=args.max_batches)
    trainer = NewImageTrainer(spec=spec, policy=policy, index=index, runtime_config={},
                              id_col=dataset.object_id_column,
                              label_col=dataset.label_column, device=args.device,
                              initialization='random')
    # The runtime model config is constructed upstream. No manual copy of its
    # metadata/patch normalization is permitted.
    model, model_config = build_model(spec, seed=args.seed, device=args.device)
    trainer.config = model_config
    request = TrainingRequest(train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            output_dir=args.out_dir, run_id='fusion_image_first_training_check',
            cycle=0, strategy='bootstrap', seed=args.seed, initial_checkpoint=None)
    result = trainer.train(request)
    print(json.dumps({'checkpoint': str(result.checkpoint_path),
                      'best_epoch': result.best_epoch,
                      'training_seconds': result.elapsed_seconds,
                      'model_source_sha': source_sha,
                      'class_names': list(CLASS_NAMES)}, indent=2))


if __name__ == '__main__':
    main()
