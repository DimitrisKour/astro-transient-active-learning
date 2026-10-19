"""Pre–Task 0 recipe extension regressions: no GPU or historical data required."""
import math
from collections import Counter

import numpy as np
import pytest

from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, label_index
from al_pipeline.astronomy.fusion_image.recipe_training_resident import (
    Recipe as OriginalRecipe, lr_multiplier as original_lr
)
from al_pipeline.astronomy.fusion_image.recipe_training_pretask0_ext import (
    Recipe, lr_multiplier, weight_vector
)


def test_legacy_80_epoch_schedule_identical():
    old = OriginalRecipe(name="old", lr_schedule="warmup_cosine", image_augmentation=True)
    new = Recipe(name="new", lr_schedule="warmup_cosine", image_augmentation=True)
    for e in range(1, 81):
        assert lr_multiplier(new, e) == pytest.approx(original_lr(old, e), abs=1e-14)


def test_short_cosine_reaches_minimum_at_epoch_25_and_stays_there():
    p = Recipe(name="short", lr_schedule="warmup_cosine", image_augmentation=True,
               cosine_decay_epochs=25)
    assert lr_multiplier(p, 1) == pytest.approx(0.1)
    assert lr_multiplier(p, 3) == pytest.approx(1.0)
    assert 0.05 < lr_multiplier(p, 10) < 1.0
    assert lr_multiplier(p, 25) == pytest.approx(0.05)
    for e in range(26, 81):
        assert lr_multiplier(p, e) == pytest.approx(0.05)


def test_invalid_horizons_rejected():
    for horizon in (2, 3, 81):
        with pytest.raises(ValueError, match="horizon"):
            Recipe(name="bad", lr_schedule="warmup_cosine", cosine_decay_epochs=horizon)


def test_effective_weights_use_sample_counts_not_object_counts():
    # One object with 3 alert rows: sample counts and object counts differ.
    rows = [
        {"object_id":"a", "label":"AGN"},
        {"object_id":"a", "label":"AGN"},
        {"object_id":"a", "label":"AGN"},
        {"object_id":"b", "label":"TDE"},
        {"object_id":"c", "label":"SN I"},
        {"object_id":"d", "label":"SN II"},
        {"object_id":"e", "label":"CV"},
    ]
    recipe=Recipe(name="eff",lr_schedule="warmup_cosine",image_augmentation=True,
                  class_weighting="effective_number",effective_count_basis="samples",
                  effective_beta=0.999)
    actual=weight_vector(rows,recipe=recipe,id_col="object_id",label_col="label")
    counts=Counter(label_index(row["label"]) for row in rows)
    ns=np.asarray([counts[i] for i in range(len(CLASS_NAMES))],dtype=float)
    expected=0.001/(1-np.power(0.999,ns))
    expected/=expected.mean()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    obj=Recipe(name="eff_object",class_weighting="effective_number",
               effective_count_basis="objects")
    assert not np.allclose(actual,weight_vector(rows,recipe=obj,id_col="object_id",
                                               label_col="label"))


def test_primary_reference_parameters_unchanged_except_selected_factor():
    short=Recipe(name="cosine_aug_short",lr_schedule="warmup_cosine",
                 image_augmentation=True,cosine_decay_epochs=25)
    eff=Recipe(name="cosine_aug_effnum",lr_schedule="warmup_cosine",
               image_augmentation=True,class_weighting="effective_number",
               effective_count_basis="samples")
    for recipe in [short,eff]:
        assert (recipe.seed,recipe.max_epochs,recipe.patience,recipe.batch_size)==(135,80,15,64)
        assert (recipe.learning_rate,recipe.weight_decay)==(1e-4,0.01)
        assert recipe.image_augmentation is True and recipe.ema is False
        assert recipe.lr_schedule == "warmup_cosine"
    assert short.class_weighting == "inverse_frequency"
    assert eff.cosine_decay_epochs is None
