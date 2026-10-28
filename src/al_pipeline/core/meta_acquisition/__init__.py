"""Model-independent adaptive mixtures of active-learning policies."""

from .tailor import DEFAULT_ARMS, TailorBandit, TailorSettings

__all__ = ["DEFAULT_ARMS", "TailorBandit", "TailorSettings"]
