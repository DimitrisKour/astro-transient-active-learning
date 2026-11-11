# Scope of copyright, licensing, and external research materials

Copyright (c) 2026 Dimitris Kourtesis.

The [GNU General Public License v3.0 (GPL-3.0-only)](LICENSE) covers the original software and accompanying documentation in this repository to the extent their copyright is held by the stated author. It does **not** grant ownership or additional permissions for any third-party code, model weights, source datasets, service credentials, or separately licensed material.

## External systems and implementations

- **AppleCiDEr, AstroMiNN, FLARE, Hyrax, BOOM and ZTF:** names in the integration code identify external scientific projects, data services, model architectures or survey resources. Their upstream code and data retain their own copyright, permissions, usage conditions and attribution requirements. This repository does not purport to relicense them.
- **Historical AstroMiNN implementation:** the astronomical adapter can import `astroMiNN_V2` from a separately installed external checkout. The upstream model implementation, its training-source repository, and its trained weights are not supplied by this package.
- **Image + metadata fusion model:** code can import `fusion_model_basics.astrominn_encoder` from a separately installed checkout. This repository's integration/experiment scripts do not imply an upstream code license or include the external model's weights.
- **Research methods:** BADGE, BALD, TAILOR, SIMILAR, DIRECT, AnchorAL and other named algorithms are research methods attributed in the relevant modules and experiment documentation; the GPLv3 license covers this repository's original implementations, not ownership of those methods or papers.
- **Third-party Python dependencies:** NumPy, PyTorch, torchvision, Astropy, scikit-learn, Matplotlib, Pandas and other optional packages are installed separately, under their own licenses.

## Recorded research outputs

The tables under `examples/data/` are aggregated or class-level **validation measurements** exported from the author's historical experiments, not raw observations, selected-object identities, training manifests, model checkpoints or external model source. The notebook figures derived from these tables are research illustrations.

The GPLv3 software license should **not** be interpreted as a new license for underlying survey data, data services, model weights or third-party content. Users who reproduce historical experiments must independently obtain access to the relevant data and code and comply with their terms. See [the export provenance and interpretation notes](examples/data/README.md).

The repository's historical HPC directory names and paths are references to non-bundled inputs, not files redistributed by the project. No endorsement by any named institution or research team is implied.

## Copyright versus scientific credit

The copyright notice identifies authorship of the repository's original code. Scientific citations to upstream models, datasets and algorithms remain appropriate independently of software copyright. For citing this repository, use [CITATION.cff](CITATION.cff).
