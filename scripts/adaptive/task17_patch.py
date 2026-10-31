"""Strict, auditable patch of verified Task11 source for Task17 unstratified L0."""
from pathlib import Path


def once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Task17 patch anchor {name}: expected once, observed {count}')
    return text.replace(old, new, 1)


def patch_source(root, overlay=None):
    root = Path(root)
    changed = []

    split = root / 'src/al_pipeline/astronomy/experiments/historical_replay.py'
    text = split.read_text(encoding='utf-8')
    text = once(text,
        '    pool_id: str = "astrominn_historical_train_replay_v1",\n',
        '    pool_id: str = "astrominn_historical_train_replay_v1",\n'
        '    selection_mode: str = "stratified",\n'
        '    initial_object_count: int | None = None,\n', 'split-signature')
    text = once(text,
        '    if not 0 < initial_fraction < 1:\n',
        '    if selection_mode not in {"stratified", "uniform_random"}:\n'
        '        raise ValueError("Unsupported L0 selection mode")\n'
        '    if selection_mode == "uniform_random" and (type(initial_object_count) is not int or initial_object_count <= 0):\n'
        '        raise ValueError("Unstratified L0 requires an exact positive object count")\n'
        '    if not 0 < initial_fraction < 1:\n', 'split-guard')
    old = '''    initial_ids: set[str] = set()
    for class_index, class_name in enumerate(MODEL_CLASS_NAMES):
        object_ids = sorted(objects_by_class[class_name])
        initial_count = max(
            minimum_initial_per_class,
            math.ceil(len(object_ids) * initial_fraction),
        )
        if len(object_ids) <= initial_count:
            raise ValueError(
                f"Class {class_name!r} has {len(object_ids)} objects; "
                f"cannot place {initial_count} in L0 and retain at "
                "least one object in U0."
            )
        generator = np.random.default_rng(np.random.SeedSequence([split_seed, class_index]))
        chosen = generator.choice(
            np.asarray(object_ids, dtype=object),
            size=initial_count,
            replace=False,
        )
        initial_ids.update(str(value) for value in chosen)
'''
    new = '''    initial_ids: set[str] = set()
    if selection_mode == "uniform_random":
        # Uniform object-level sampling from the full historical population.
        # No labels, stratification, minimum class counts, or rejection sampling
        # enter the RNG process. Sorted IDs make selection independent of row order.
        all_ids = np.asarray(sorted(canonical_label_by_object), dtype=object)
        if initial_object_count >= len(all_ids):
            raise ValueError("L0 must leave at least one candidate in U0")
        generator = np.random.default_rng(np.random.SeedSequence([split_seed, 17]))
        initial_ids = set(str(v) for v in generator.choice(
            all_ids, size=initial_object_count, replace=False))
    else:
        for class_index, class_name in enumerate(MODEL_CLASS_NAMES):
            object_ids = sorted(objects_by_class[class_name])
            initial_count = max(
                minimum_initial_per_class,
                math.ceil(len(object_ids) * initial_fraction),
            )
            if len(object_ids) <= initial_count:
                raise ValueError(
                    f"Class {class_name!r} has {len(object_ids)} objects; "
                    f"cannot place {initial_count} in L0 and retain at "
                    "least one object in U0."
                )
            generator = np.random.default_rng(np.random.SeedSequence([split_seed, class_index]))
            chosen = generator.choice(
                np.asarray(object_ids, dtype=object),
                size=initial_count,
                replace=False,
            )
            initial_ids.update(str(value) for value in chosen)
'''
    text = once(text, old, new, 'split-selection')
    text = once(text,
        '        "design": "historical_train_object_stratified_replay",\n',
        '        "design": "historical_train_object_uniform_random_replay" if selection_mode == "uniform_random" else "historical_train_object_stratified_replay",\n'
        '        "initial_selection_mode": selection_mode,\n'
        '        "initial_object_count_requested": initial_object_count,\n', 'split-provenance')
    split.write_text(text, encoding='utf-8')
    changed.append(str(split.relative_to(root)))

    runner = root / 'scripts/run_astrominn_closed_loop.py'
    text = runner.read_text(encoding='utf-8')
    text = once(text,
        '    parser.add_argument("--initial-fraction", type=float, default=0.10)\n',
        '    parser.add_argument("--initial-fraction", type=float, default=0.10)\n'
        '    parser.add_argument("--l0-selection", choices=("uniform_random",), required=True)\n'
        '    parser.add_argument("--l0-count", type=int, required=True)\n', 'runner-flags')
    text = once(text,
        '    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n'
        '        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n',
        '    if set(args.strategies) - {"random", "entropy"}:\n'
        '        raise ValueError("Task17 allows random and entropy only")\n', 'runner-lock')
    text = once(text,
        '        pool_id=f"{args.suite_id}_historical_u0",\n',
        '        pool_id=f"{args.suite_id}_historical_u0",\n'
        '        selection_mode=args.l0_selection,\n'
        '        initial_object_count=args.l0_count,\n', 'runner-split-call')
    text = once(text,
        '        "minimum_initial_per_class": args.minimum_initial_per_class,\n',
        '        "minimum_initial_per_class": None,\n'
        '        "initial_selection_mode": args.l0_selection,\n'
        '        "initial_object_count_requested": args.l0_count,\n', 'runner-provenance')
    text = once(text,
        '    if args.bootstrap_only and not args.run:\n',
        '    if args.l0_count != 476:\n'
        '        raise ValueError("Task17 L0 size locked to 476 objects")\n'
        '    if args.bootstrap_only and not args.run:\n', 'runner-count-lock')
    text = once(text,
        '    if not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")\n',
        '    if not args.require_existing_bootstrap and not args.bootstrap_only and args.run:\n'
        '        raise ValueError("Task17 cycles must reuse a new completed Task17 bootstrap")\n',
        'runner-new-bootstrap')
    runner.write_text(text, encoding='utf-8')
    changed.append(str(runner.relative_to(root)))

    # Truly random L0 can omit a class. We must not reject/resample based on
    # candidate labels to fix that; allow training with the observed classes.
    trainer = root / 'src/al_pipeline/astronomy/astrominn/training.py'
    text = trainer.read_text(encoding='utf-8')
    text = once(text,
        '''    missing = [MODEL_CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(
            f"Manifest-derived class weighting requires every class in L_t; missing {missing}."
        )

    raw = torch.tensor(
        [count ** (-policy.class_weight_exponent) for count in counts],
''',
        '''    missing = [MODEL_CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
    # Task17 only: preserve genuinely unstratified sampling even when L0
    # contains no examples of a class. Smooth absent counts to one for the
    # weight calculation; the class has no loss term until examples arrive.
    safe_counts = [max(count, 1) for count in counts]

    raw = torch.tensor(
        [count ** (-policy.class_weight_exponent) for count in safe_counts],
''', 'zero-class-trainer')
    text = once(text,
        '        "normalization": "unit_mean",\n',
        '        "normalization": "unit_mean",\n'
        '        "zero_count_classes": missing,\n'
        '        "zero_count_smoothing": "max(sample_count,1) for weights only",\n', 'trainer-metadata')
    trainer.write_text(text, encoding='utf-8')
    changed.append(str(trainer.relative_to(root)))
    return changed
