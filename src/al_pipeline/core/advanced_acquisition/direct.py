"""DIRECT: per-class score ordering, batched VReduce and boundary acquisition.

Adapted to an object-level pool and exact per-cycle budgets. The oracle is a
callback invoked ONLY after a subbatch is selected; no candidate truth appears
in the selector's inputs. Consult Nuggehalli et al. ICML 2025 Algorithms 1/2.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class DirectInput:
    candidate_ids: tuple[str, ...]
    probabilities: np.ndarray
    confirmed_ids: tuple[str, ...]
    confirmed_probabilities: np.ndarray
    confirmed_labels: tuple[str, ...]
    class_names: tuple[str, ...]

    def __post_init__(self):
        ids = tuple(self.candidate_ids)
        anchors = tuple(self.confirmed_ids)
        classes = tuple(self.class_names)
        if not ids or len(set(ids)) != len(ids) or not anchors or len(set(anchors)) != len(anchors):
            raise ValueError('Object IDs must be unique and nonempty')
        if set(ids) & set(anchors): raise ValueError('Candidate and confirmed IDs must be disjoint')
        if len(classes)<2 or len(set(classes)) != len(classes):raise ValueError('Invalid classes')
        if len(self.confirmed_labels) != len(anchors) or set(self.confirmed_labels)-set(classes):
            raise ValueError('Confirmed labels must align with anchors and classes')
        for name, values, rows in [('probabilities',self.probabilities,len(ids)),
                                   ('confirmed_probabilities',self.confirmed_probabilities,len(anchors))]:
            p=np.asarray(values,dtype=np.float64)
            if p.shape!=(rows,len(classes)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1,atol=1e-6):
                raise ValueError(f'Invalid {name} shape or normalization')
            object.__setattr__(self,name,p)
        object.__setattr__(self,'candidate_ids',ids)
        object.__setattr__(self,'confirmed_ids',anchors)
        object.__setattr__(self,'class_names',classes)
        object.__setattr__(self,'confirmed_labels',tuple(self.confirmed_labels))


def _margins(p):
    """Paper eq. (2): p_k - max_j p_j, including j=k; tie by p_k descending."""
    return p - p.max(axis=1,keepdims=True)


def _allocation(total, classes, order):
    if total<0:raise ValueError('Negative budget')
    res=np.full(classes,total//classes,dtype=int)
    for k in order[:total%classes]:res[int(k)]+=1
    return res


def _losses(observed, target, n):
    """Threshold loss: positive class predicted on RIGHT, rest on LEFT."""
    obs=np.zeros(n,dtype=np.int8)
    for position,label in observed.items():
        obs[int(position)] = 1 if label==target else -1
    # errors left: positive, errors right: negative
    positive=(obs==1).astype(int)
    negative=(obs==-1).astype(int)
    return np.r_[0,np.cumsum(positive)] + (negative.sum()-np.r_[0,np.cumsum(negative)])


def _version_interval(observed,target,n):
    """Conservative version-space start from confirmed extremes (not truth in U)."""
    positive=[i for i,y in observed.items() if y==target]
    negative=[i for i,y in observed.items() if y!=target]
    low=max((i+1 for i in negative if not positive or i<min(positive)),default=0)
    high=min((i for i in positive if not negative or i>max(negative)),default=n)
    if low>=high:return 0,n  # noisy / overlapping: retain all hypotheses
    return low,high


def _shrink_interval(observed,target,n,low,high,new_width):
    """Algorithm 1: minimize maximum endpoint empirical threshold loss."""
    width=max(1,min(high-low,int(new_width)))
    loss=_losses(observed,target,n)
    starts=np.arange(low,high-width+1,dtype=int)
    objective=np.maximum(loss[starts],loss[starts+width])
    # Deterministic tie break closest to previous center.
    center=(low+high)/2
    winner=int(np.lexsort((starts,np.abs(starts+width/2-center),objective))[0])
    return int(starts[winner]),int(starts[winner]+width)


def _optimal_threshold(observed,target,n):
    loss=_losses(observed,target,n)
    best=np.flatnonzero(loss==loss.min())
    # Paper: choose the threshold closest to the midpoint in a tie.
    return int(best[np.argmin(np.abs(best-n/2))])


class DirectSelector:
    """VReduce label queries + nearest learned one-vs-rest thresholds.

    query(ids, *, phase, class_name, subround) returns a tuple of labels in the
    same order, from a provider that exposes ONLY requested candidate labels.
    """
    def __init__(self,data:DirectInput,*,parallel_batch:int=5,seed:int=135):
        if not isinstance(data,DirectInput):raise TypeError('Expected DirectInput')
        if type(parallel_batch) is not int or parallel_batch<1:raise ValueError('Bad parallel batch')
        self.data,self.parallel_batch,self.seed=data,parallel_batch,int(seed)

    def select(self,*,budget:int,query:Callable):
        from al_pipeline.core.types import SelectionBatch
        d=self.data
        n=len(d.candidate_ids);c=len(d.class_names)
        if type(budget) is not int or not 1<=budget<=n:raise ValueError('Invalid budget')
        all_ids=d.confirmed_ids+d.candidate_ids
        all_p=np.concatenate((d.confirmed_probabilities,d.probabilities))
        margins=_margins(all_p)
        rng=np.random.default_rng(self.seed)
        class_order=rng.permutation(c)
        phase1=_allocation(budget//2,c,class_order)
        phase2=_allocation(budget-int(phase1.sum()),c,class_order)
        seen=dict(zip(d.confirmed_ids,d.confirmed_labels,strict=True))
        remaining=set(d.candidate_ids)
        picked=[];steps=[];rounds=[];thresholds={}
        rankings={}
        for k in class_order:
            k=int(k);class_name=d.class_names[k]
            # Stable order: margin ascending, p_k descending, then original index.
            order=np.lexsort((np.arange(len(all_ids)), -all_p[:,k], margins[:,k]))
            sorted_ids=[all_ids[int(i)] for i in order]
            pos={oid:i for i,oid in enumerate(sorted_ids)}
            n_total=len(sorted_ids)
            known=lambda: {pos[oid]:label for oid,label in seen.items() if oid in pos}
            low,high=_version_interval(known(),class_name,n_total)
            n_queries=int(phase1[k]);m=(n_queries+self.parallel_batch-1)//self.parallel_batch
            initial_width=max(1,high-low)
            factor=initial_width**(1/max(m,1))
            completed=0
            for iteration in range(m):
                q=min(self.parallel_batch,n_queries-completed,len(remaining))
                if q==0:break
                available=[oid for oid in sorted_ids[low:high] if oid in remaining]
                if len(available)<q:
                    # If the interval is exhausted, expand nearest to its midpoint.
                    available=sorted(remaining,key=lambda oid:(abs(pos[oid]-(low+high)/2),pos[oid]))
                selected=[str(x) for x in rng.choice(np.asarray(available,dtype=object),size=q,replace=False)]
                labels=tuple(query(tuple(selected),phase='vreduce',class_name=class_name,subround=iteration+1))
                if len(labels)!=q or set(labels)-set(d.class_names):raise ValueError('Oracle returned invalid labels')
                for oid,label in zip(selected,labels,strict=True):
                    if oid not in remaining or oid in seen:raise ValueError('Duplicate oracle request')
                    seen[oid]=label;remaining.remove(oid);picked.append(oid);steps.append('vreduce')
                completed+=q
                target=max(1,int(round(initial_width/(factor**(iteration+1)))))
                low,high=_shrink_interval(known(),class_name,n_total,low,high,target)
                rounds.append({'phase':'vreduce','class':class_name,'subround':iteration+1,
                               'count':q,'interval':[low,high],'queried_ids':selected})
            rankings[k]=(pos,n_total)
            thresholds[class_name]={'version_interval':[low,high]}
        # Phase 2 begins ONLY after VReduce has finished for all classes.
        for k in class_order:
            k=int(k);class_name=d.class_names[k]
            pos,n_total=rankings[k]
            sorted_ids=sorted(pos,key=pos.__getitem__)
            known={pos[oid]:label for oid,label in seen.items() if oid in pos}
            threshold=_optimal_threshold(known,class_name,n_total)
            thresholds[class_name].update({'rank':threshold,'total_ranked':n_total})
            q=min(int(phase2[k]),len(remaining))
            # Equal distance in index order to threshold, deterministic tie by rank.
            neighbors=sorted((oid for oid in remaining),key=lambda oid:(abs(pos[oid]-threshold),pos[oid]))
            final=neighbors[:q]
            if final:
                labels=tuple(query(tuple(final),phase='boundary',class_name=class_name,subround=1))
                if len(labels)!=len(final) or set(labels)-set(d.class_names):raise ValueError('Invalid boundary oracle labels')
                for oid,label in zip(final,labels,strict=True):
                    if oid not in remaining or oid in seen:raise ValueError('Duplicate boundary acquisition')
                    seen[oid]=label;remaining.remove(oid);picked.append(oid);steps.append('boundary')
                rounds.append({'phase':'boundary','class':class_name,'count':len(final),'queried_ids':final})
        if len(picked)!=budget:raise RuntimeError(f'DIRECT acquired {len(picked)} rather than {budget}')
        indexes={oid:i for i,oid in enumerate(d.candidate_ids)}
        return SelectionBatch(ids=picked,indices=np.asarray([indexes[x] for x in picked]),
            scores=np.asarray([float(budget-i) for i in range(len(picked))]),
            metadata={'selector':'direct_vreduce','score_semantics':'selection_order_priority_not_model_uncertainty','phase1_budget':int(phase1.sum()),
                      'phase2_budget':int(phase2.sum()),'parallel_batch':self.parallel_batch,
                      'class_order':[d.class_names[int(k)] for k in class_order],
                      'thresholds':thresholds,'steps':steps,'subrounds':rounds,
                      'known_candidate_labels_used_only_after_query':True,
                      'margin':'p_k - max_all_classes(p); rank ties p_k descending',
                      'rounding':'floor half to VReduce; distribute per-class remainders by seeded class permutation'})
