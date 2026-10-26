"""Projected expected-Fisher A-optimal greedy acquisition (BAIT-inspired).

NOT an implementation of full-parameter BAIT or its forward-backward procedure.
One sketch of last-layer class gradients is used for pool, labelled anchors,
and candidate marginal gains. Neither ground-truth candidate labels nor held-out
validation/test data are consulted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FisherBatchInput:
    candidate_ids: tuple[str, ...]
    probabilities: np.ndarray
    embeddings: np.ndarray
    labeled_probabilities: np.ndarray
    labeled_embeddings: np.ndarray
    rank: int = 32
    ridge: float = 1.0
    projection_seed: int = 271828
    _engine: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.candidate_ids = tuple(self.candidate_ids)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('Fisher candidate IDs must be nonempty and unique')
        for name in ('probabilities','embeddings','labeled_probabilities','labeled_embeddings'):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1 or not np.isfinite(value).all():
                raise ValueError(f'{name} must be finite nonempty 2-D array')
            setattr(self, name, value)
        p, h = self.probabilities, self.embeddings
        lp, lh = self.labeled_probabilities, self.labeled_embeddings
        if p.shape[0] != len(self.candidate_ids) or lp.shape[0] != lh.shape[0] or p.shape[1] != lp.shape[1] or h.shape[1] != lh.shape[1]:
            raise ValueError('Fisher row, feature, or class alignment error')
        if p.shape[1] < 2 or np.any(p < 0) or np.any(lp < 0) or not np.allclose(p.sum(axis=1),1,atol=1e-6) or not np.allclose(lp.sum(axis=1),1,atol=1e-6):
            raise ValueError('Fisher requires properly normalized multiclass probabilities')
        if not 2 <= self.rank <= 256 or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('Invalid Fisher rank/ridge')

    @property
    def engine(self):
        if self._engine is None:
            self._engine = ProjectedFisherAOptimal(self)
        return self._engine


class ProjectedFisherAOptimal:
    """Forward greedy pool-risk A-optimality using Woodbury rank-C updates.

    Projected expected Fisher F_i = Z_i.T @ Z_i; each Z_i has C rows,
    with z_(i,y) = sqrt(p_y) P^T[(e_y-p) ⊗ (h_i,1)].
    Objective: trace(F_pool_mean @ inverse(ridge I + F_L + F_selected)).
    This is explicitly an approximation in the *fixed* projected gradient space.
    """
    def __init__(self, inp: FisherBatchInput):
        self.data = inp
        h = np.vstack((inp.embeddings, inp.labeled_embeddings))
        scale = float(np.sqrt(np.mean(h*h)))
        if scale == 0:
            scale = 1.0
        features = np.c_[h / scale, np.ones(len(h))]
        classes = inp.probabilities.shape[1]
        rng = np.random.default_rng(inp.projection_seed)
        projection = rng.normal(size=(classes,features.shape[1],inp.rank)) / np.sqrt(inp.rank)
        # Exact projection of each class-conditioned last-layer pseudo-gradient.
        q = np.einsum('nd,cdr->ncr',features,projection,optimize=True)
        p = np.vstack((inp.probabilities,inp.labeled_probabilities))
        q -= np.einsum('nc,ncr->nr',p,q,optimize=True)[:,None,:]
        q *= np.sqrt(p)[:,:,None]
        n = len(inp.candidate_ids)
        self.z = q[:n]
        self.labeled = q[n:]
        self.target = np.einsum('ncr,ncs->rs',self.z,self.z,optimize=True) / n
        information = np.einsum('ncr,ncs->rs',self.labeled,self.labeled,optimize=True)
        information.flat[::inp.rank+1] += inp.ridge
        self.inverse = np.linalg.inv(information)
        self.inverse = (self.inverse+self.inverse.T)*0.5
        self._initial = None

    def _gain(self, inverse: np.ndarray) -> np.ndarray:
        q = np.matmul(self.z,inverse)
        a = np.matmul(q,np.swapaxes(self.z,1,2))
        a[:,np.arange(a.shape[1]),np.arange(a.shape[1])] += 1.
        b = np.matmul(np.matmul(q,self.target),np.swapaxes(q,1,2))
        gain = np.trace(np.linalg.solve(a,b),axis1=1,axis2=2)
        if not np.all(np.isfinite(gain)):
            raise FloatingPointError('Nonfinite Fisher marginal gain')
        return np.maximum(gain,0.)

    def initial_gains(self) -> np.ndarray:
        if self._initial is None:
            self._initial = self._gain(self.inverse)
        return self._initial.copy()

    def select(self, k: int) -> tuple[np.ndarray,np.ndarray]:
        if not 0 <= k <= len(self.z):
            raise ValueError('Invalid Fisher acquisition budget')
        inverse = self.inverse.copy()
        used = np.zeros(len(self.z),dtype=bool)
        chosen, values = [], []
        for step in range(k):
            gains = self._gain(inverse) if step else self.initial_gains()
            gains[used] = -np.inf
            row = int(np.argmax(gains))
            chosen.append(row)
            values.append(float(gains[row]))
            used[row] = True
            z = self.z[row]
            mid = np.eye(z.shape[0]) + z@inverse@z.T
            update = inverse@z.T@np.linalg.solve(mid,z@inverse)
            inverse -= update
            inverse = (inverse+inverse.T)*0.5
        return np.asarray(chosen,dtype=np.int64),np.asarray(values,dtype=float)


class FisherBatchScorer:
    def score(self, data: FisherBatchInput) -> np.ndarray:
        return data.engine.initial_gains()


class FisherBatchSelector:
    def __init__(self, data: FisherBatchInput):
        self.data = data

    def select(self, *, ids, scores, k, pool):
        from al_pipeline.core.types import SelectionBatch
        from al_pipeline.core.advanced_acquisition._common import eligible_candidates,validate_scores
        ids_arr,values,eligible = eligible_candidates(ids,scores,k=k,pool=pool,expected_rows=len(self.data.candidate_ids))
        if tuple(ids_arr) != self.data.candidate_ids:
            raise ValueError('Fisher IDs not aligned')
        validate_scores(values,self.data.engine.initial_gains(),method='Fisher batch')
        if len(eligible) != len(ids_arr):
            raise ValueError('Fisher selection expects precisely current U_t')
        chosen,gains = self.data.engine.select(k)
        return SelectionBatch(ids=ids_arr[chosen].tolist(),indices=chosen,scores=gains,
                              metadata={'selector':'projected_fisher_forward_a_optimal',
                                        'rank':self.data.rank,'ridge':self.data.ridge,
                                        'projection_seed':self.data.projection_seed,
                                        'labeled_anchor_count':len(self.data.labeled_embeddings),
                                        'score_semantics':'dynamic_trace_reduction',
                                        'full_bait':False})
