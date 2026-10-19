"""Task-1 object-level acquisition from cached, UNLABELED candidate stages.

Only object IDs, stage IDs, logits and optional embeddings are accepted here.
Historical target labels are never passed into this module. Selectors are the
existing, tested model-independent AL selectors; no bespoke diversity policy.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.scoring.random import RandomScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import ClassificationPredictions

POLICIES = ('random', 'entropy', 'entropy_diversity', 'entropy_balanced_explore')
PHASE1 = ('random', 'entropy')
PHASE2 = ('entropy_diversity', 'entropy_balanced_explore')


def acquisition_seed(*, seed: int, cycle: int) -> int:
    if seed < 0 or not 1 <= cycle <= 9:
        raise ValueError('Invalid acquisition seed/cycle')
    return seed * 1009 + cycle * 10007


def choose(*, policy: str, ids: list[str], logits: np.ndarray,
           embeddings: np.ndarray | None, k: int, seed: int, cycle: int) -> dict:
    """Select exactly k objects. No label arguments or oracle are available here.

    Diversity is the already-tested 3x entropy shortlist + farthest-first;
    'balanced_explore' is the historical **soft exploration** method with
    20% uniform exploration, NOT a hidden-label class-quota scheme.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown acquisition policy: {policy}')
    ids = list(ids)
    if len(ids) != len(set(ids)) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise ValueError('Duplicate/empty candidate object ID')
    x = ClassificationPredictions(np.asarray(logits), 'logits')
    if x.values.shape != (len(ids), 5):
        raise ValueError('Expected one five-class logit vector per candidate object')
    if not 1 <= k <= len(ids):
        raise ValueError('Invalid acquisition budget')
    rng = np.random.default_rng(acquisition_seed(seed=seed, cycle=cycle))
    pool = Pool.from_ids(ids)
    entropy = EntropyScorer().score(x)
    scores = RandomScorer(rng).score(len(ids)) if policy == 'random' else entropy
    if policy == 'entropy_diversity':
        if embeddings is None or np.asarray(embeddings).shape[0] != len(ids):
            raise ValueError('Diversity requires embeddings for all candidates')
        selector = DiversitySelector(np.asarray(embeddings), shortlist_multiplier=3.0)
    elif policy == 'entropy_balanced_explore':
        selector = ExplorationSelector(exploration_fraction=0.2, rng=rng)
    else:
        selector = TopKSelector()
    batch = selector.select(ids=ids, scores=scores, k=k, pool=pool)
    if len(batch.ids) != k or len(set(batch.ids)) != k or set(batch.ids) - set(ids):
        raise RuntimeError('Selection is not a set of k eligible unique objects')
    probs = x.as_probabilities()
    pred = np.argmax(probs, axis=1)
    chosen = batch.indices.astype(int)
    return {
        'policy':policy, 'seed':seed, 'cycle':cycle,
        'acquisition_rng_seed':acquisition_seed(seed=seed, cycle=cycle),
        'selected_ids':list(batch.ids),
        'selected_indices':chosen.tolist(),
        'selected_scores':batch.scores.tolist(),
        'selected_entropy':[float(entropy[i]) for i in chosen],
        'selected_predicted_class':[int(pred[i]) for i in chosen],
        'selected_class_composition':{str(i):int(Counter(pred[chosen])[i]) for i in range(5)},
        'pool_class_composition':{str(i):int(Counter(pred)[i]) for i in range(5)},
        'selector_metadata':batch.metadata,
        'candidate_count':len(ids),
    }
