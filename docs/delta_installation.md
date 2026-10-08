# Delta installation

The extraction patch is applied to the empty sibling `active-learning` repository.
The old checkout, upstream clones, datasets, and running experiments are preserved.

## Environment and editable installs

Clone the working environment so the current experiments keep their dependencies:

```bash
conda create --name al-pipeline --clone applecider
conda activate al-pipeline
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning

# Guard against running package changes in the original environment.
test "$CONDA_DEFAULT_ENV" = al-pipeline

# Remove the old fork's editable package from this cloned environment only.
python -m pip uninstall -y applecider
python -m pip install --no-deps --no-build-isolation -e ../applecider
python -m pip install --no-deps --no-build-isolation -e .

python -c 'import al_pipeline; print(al_pipeline.__file__)'
python -c 'import applecider; print(applecider.__file__)'
python -m pytest -q
```

If pytest is absent, install it in the cloned environment. The extraction itself
does not require reinstalling or upgrading PyTorch, Hyrax, or FLARE. FLARE can
be installed from `../flare` when the corresponding notebooks are migrated.

`configs/delta.toml` is ignored by Git. The bundle's apply script can copy the
existing local registry into that path without changing its contents. Review
that it still points to the current HDD/NVMe locations and historical artifacts.
The new example config is documentation, not a replacement for the live registry.

## Launchers

First check imports and argument parsing:

```bash
python scripts/run_astrominn_simulated_al.py --help
python scripts/run_astrominn_historical_replay.py --help
python scripts/report_astrominn_multiseed.py --help
```

Reuse each experiment's current launch arguments with two explicit edits:
`--registry configs/delta.toml` and a fresh `--suite-id` for extraction checks.
The documented A/B launcher names and options are unchanged. Inspect the
preflight before adding `--run` in a Slurm compute job. Do not reuse an existing
run directory for a trial. B preflight writes its split under the new suite ID;
it does not train until `--run` is passed.

After verification, commit the extracted repository and choose its GitHub remote.
No GitHub repository or remote is created by the extraction bundle.
