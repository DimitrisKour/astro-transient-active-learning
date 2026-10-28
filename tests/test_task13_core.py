"""CPU-only regression checks for Task13's published TAILOR control."""

from __future__ import annotations

import unittest

import numpy as np

from al_pipeline.core.advanced_acquisition.fisher_batch import (
    FisherBatchInput,
    FisherBatchScorer,
    FisherBatchSelector,
)
from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector
from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig
from al_pipeline.core.meta_acquisition.mixed_selection import select_mixed_batch
from al_pipeline.core.meta_acquisition.tailor import TailorBandit, TailorSettings
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.class_balanced import ClassBalancedInput
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.types import ClassificationPredictions


class TestTailorController(unittest.TestCase):
    def test_original_inverse_count_reward(self):
        b = TailorBandit(class_names=("AGN", "CV", "SN I", "SN II", "TDE"))
        weights = b.reward_weights([87, 19, 296, 69, 5])
        np.testing.assert_allclose(weights, [1 / 435, 1 / 95, 1 / 1480, 1 / 345, 1 / 25])
        self.assertAlmostEqual(weights[-1] / weights[2], 296 / 5)

    def test_posterior_uses_only_assigned_revealed_labels_and_discount(self):
        b = TailorBandit(class_names=("SN I", "TDE"), settings=TailorSettings())
        b.update(
            ["entropy_diversity", "fisher_batch", "entropy_diversity"],
            ["TDE", "SN I", "SN I"],
        )
        np.testing.assert_allclose(b.alpha, [
            [1.9, 1.9],
            [0.9, 0.9],
            [1.9, 0.9],
        ])
        self.assertEqual(b.rounds_seen, 1)
        self.assertRaises(ValueError, b.update, ["fisher_batch"], ["not_a_class"])

    def test_uniform_mixture_is_seeded_and_independent_of_class_counts(self):
        b = TailorBandit(class_names=("SN I", "TDE"))
        a = b.choose(batch_size=100, confirmed_counts=[100, 1],
                     method="uniform_mixture", rng=np.random.default_rng(123))
        c = b.choose(batch_size=100, confirmed_counts=[1, 100],
                     method="uniform_mixture", rng=np.random.default_rng(123))
        self.assertEqual(a, c)
        self.assertEqual(set(a), set(b.settings.arms))

    def test_tailor_prefers_known_rare_class_discoverer(self):
        b = TailorBandit(class_names=("SN I", "TDE"))
        b.alpha = np.asarray([[101., 1.], [1., 101.], [50., 50.]])
        allocations = b.choose(
            batch_size=200,
            confirmed_counts=[200, 5],
            method="tailor_diversity",
            rng=np.random.default_rng(42),
        )
        self.assertGreater(allocations.count("entropy_balanced_explore"), 195)


class TestMixedSelection(unittest.TestCase):
    def setUp(self):
        gen = np.random.default_rng(321)
        self.ids = tuple(f"obj_{i:03d}" for i in range(40))
        self.pred = ClassificationPredictions(gen.dirichlet([2, 1, 3], size=40), "probabilities")
        self.emb = gen.normal(size=(40, 7))
        self.labeled_prob = gen.dirichlet([2, 1, 3], size=12)
        self.labeled_emb = gen.normal(size=(12, 7))
        self.counts = np.asarray([12, 3, 40])
        self.policy = BatchAcquisitionConfig()

    def mixed(self, assignments, seed=88):
        return select_mixed_batch(
            object_ids=self.ids,
            predictions=self.pred,
            embeddings=self.emb,
            confirmed_counts=self.counts,
            labeled_probabilities=self.labeled_prob,
            labeled_embeddings=self.labeled_emb,
            arm_assignments=assignments,
            exploration_rng=np.random.default_rng(seed),
            batch_policy=self.policy,
        )

    def test_mixed_selection_unique_deterministic_aligned(self):
        arms = ("entropy_diversity", "entropy_balanced_explore", "fisher_batch") * 4
        a = self.mixed(arms)
        b = self.mixed(arms)
        self.assertEqual(a.ids, b.ids)
        self.assertEqual(a.metadata["arm_assignments"], list(arms))
        self.assertEqual(len(set(a.ids)), 12)
        np.testing.assert_array_equal(a.indices, b.indices)
        self.assertTrue(np.all(np.isfinite(a.scores)))

    def test_entire_batch_entropy_diversity_matches_original_selector(self):
        k = 12
        from al_pipeline.core.scoring.entropy import EntropyScorer
        entropy = EntropyScorer().score(self.pred)
        original = DiversitySelector(self.emb, shortlist_multiplier=3).select(
            ids=self.ids, scores=entropy, k=k, pool=Pool.from_ids(self.ids),
        )
        mixed = self.mixed(("entropy_diversity",) * k)
        self.assertEqual(mixed.ids, original.ids)

    def test_entire_batch_balanced_explore_matches_original_selector(self):
        k = 12
        balanced = self.policy.make_scorer().score(
            ClassBalancedInput(self.pred, self.counts)
        )
        original = ExplorationSelector(
            exploration_fraction=0.2, rng=np.random.default_rng(88),
        ).select(ids=self.ids, scores=balanced, k=k, pool=Pool.from_ids(self.ids))
        mixed = self.mixed(("entropy_balanced_explore",) * k)
        self.assertEqual(mixed.ids, original.ids)

    def test_entire_batch_fisher_matches_original_selector(self):
        k = 12
        data = FisherBatchInput(
            candidate_ids=self.ids,
            probabilities=self.pred.as_probabilities(),
            embeddings=self.emb,
            labeled_probabilities=self.labeled_prob,
            labeled_embeddings=self.labeled_emb,
        )
        original = FisherBatchSelector(data).select(
            ids=self.ids, scores=FisherBatchScorer().score(data),
            k=k, pool=Pool.from_ids(self.ids),
        )
        mixed = self.mixed(("fisher_batch",) * k)
        self.assertEqual(mixed.ids, original.ids)

    def test_unknown_arm_fails_closed(self):
        with self.assertRaises(ValueError):
            self.mixed(("unknown",))


if __name__ == "__main__":
    unittest.main()
