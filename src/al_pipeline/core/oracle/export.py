"""Utilities for exporting active-learning queries to external label providers.

The concrete oracle workflow has not yet been chosen.

TODO
----
Determine with supervisors how selected objects will be submitted for
inspection or labeling.

Questions to resolve include:

1. What is the label provider?
   - human inspection
   - Fritz
   - spectroscopic follow-up
   - another external service/workflow

2. Which information should accompany each selected object?
   Possible fields include:
   - stable object ID
   - acquisition score
   - predicted class
   - class probabilities
   - acquisition round
   - model/checkpoint identifier
   - relevant metadata
   - image/thumbnail references

3. What export format is appropriate?
   - CSV
   - Parquet
   - JSON
   - direct API submission
   - another format

4. Is labeling synchronous or asynchronous?

5. How should returned labels be imported and matched to object IDs?

6. How should unresolved or failed queries be represented?
   These may remain pending or be released back to the unlabeled pool.

7. Determine what provenance information is required for reproducible
   active-learning experiments.

Do not implement a concrete export format until the oracle workflow is known.
"""
