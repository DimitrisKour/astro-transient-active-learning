# Pre-publication review

This repository is intended to be made public **only after** a source and Git-history audit. A clean working tree and a clean default-branch snapshot do not establish that all historical commits, other branches, GitHub Actions logs, or attached artifacts are safe.

## Required checks

1. From a fresh full clone with all remote branches and tags fetched, run `python scripts/publication_audit.py --history`. The scanner reports only file names and detection rules, not secret values.
2. Inspect `git branch -a`, `git tag -l`, commits and merged pull requests for old private material. Review any workflow logs and artifacts that will be visible after the visibility change.
3. Run `python -m pytest -q` and the CPU teaching checks. Optional astronomical/HPC experiments have independent external dependencies.
4. Verify that externally referenced AstroMiNN, AppleCiDEr, FLARE and fusion sources, checkpoints, raw datasets and credentials remain outside the repository. Inspect recorded research exports for redistributable content.
5. Rotate **real** leaked credentials before attempting history cleanup. A new commit that deletes a secret is insufficient: the old blob remains in history. If cleanup is required, use a history-rewriting tool or a fresh clean public repository, and re-audit every ref.
6. Check the [license](../LICENSE), [third-party scope](../THIRD_PARTY.md), and [citation metadata](../CITATION.cff) against the intended release.

The history scanner is deliberately conservative, not proof of legal title or security. Paths to research environments are references, not credentials, but avoid sharing active access details, unpublished findings or protected research data in issues, logs, and commit messages.

**Visibility is a separate action**. No test, pull request, or license commit makes a private repository public.
