"""Single-file NVMe stage cache for independent short scientific training jobs.

Immutable, label-free materialization of validated v2 shards.  Validation labels
stay with caller-owned manifests; all historic training rows can be indexed for
closed-loop AL without exposing their labels to acquisition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import label_index, sha256, write_json_atomic

SOURCE = "historical_full_train_plus_fixed_validation"
FORMAT = 1


def read_identity(source: Path) -> tuple[dict, dict]:
    source = Path(source)
    final = json.loads((source / "cache_manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if final.get("source") != SOURCE or final.get("cache_version") != 2:
        raise ValueError("Expected finalized complete historical v2 stage cache")
    if final.get("labels_saved_in_cache") is not False:
        raise ValueError("Refusing label-bearing source cache")
    if final.get("plan_sha256") != sha256(source / "plan.json"):
        raise ValueError("Source plan integrity mismatch")
    if final.get("manifest_sha256") != plan["manifest_sha256"]:
        raise ValueError("Source split integrity mismatch")
    if final.get("manifest_rows") != {"train": 29161, "validation": 14148}:
        raise ValueError("Not the expected complete historical split")
    if final.get("unique_objects") != 7094 or final.get("unique_stages") != 43309:
        raise ValueError("Expected 7,094 historical objects and 43,309 stages")
    if len(final["shards"]) != plan["n_shards"]:
        raise ValueError("Incomplete shard manifest")
    return final, plan


def build(*, source: Path, target: Path, batch_rows: int = 384) -> dict:
    """Buffered shard-to-file consolidation; no writable or source memory mappings.

    Each output is an ordinary sequentially written NPY stream with a fixed
    header. Unlike ``open_memmap(..., mode="w+")``, the writer does not touch
    filesystem-backed pages via mmap, so an I/O failure can raise an OSError
    with the relevant shard and file rather than killing Python with SIGBUS.
    """
    source, target = Path(source), Path(target)
    final, plan = read_identity(source)
    if target.exists():
        raise FileExistsError(f"Target exists; refusing overwrite: {target}")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    count = int(final["unique_stages"])
    image_shape = (count, 3, 63, 63)
    meta_shape = (count, 55)
    anticipated = 4 * count * (3*63*63 + 55)
    target.parent.mkdir(parents=True, exist_ok=True)
    # df reports cluster capacity, not project quotas; this is a heuristic only.
    available = shutil.disk_usage(target.parent).free
    if available < anticipated + 256 * 1024 * 1024:
        raise OSError(f"Insufficient free space for {anticipated} bytes of cache arrays")
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.building_", dir=target.parent))
    print(f"[fast-cache] buffered build started source={source} temp={tmp} "
          f"bytes_expected={anticipated} free_bytes={available}", flush=True)
    started = time.perf_counter()
    written = 0
    seen: set[tuple[str, str, int]] = set()
    per_split = {"train": 0, "validation": 0}
    try:
        # Writing a NumPy header before the data preserves NPY compatibility.
        # The files are *fully allocated by normal writes* rather than mmap.
        with (tmp / "images.npy").open("wb", buffering=1024*1024) as images_out, \
             (tmp / "metadata.npy").open("wb", buffering=1024*1024) as meta_out, \
             (tmp / "stage_index.csv").open("w", encoding="utf-8", newline="") as index_out:
            dtype = np.dtype("<f4")
            np.lib.format.write_array_header_2_0(images_out, {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False, "shape": image_shape})
            np.lib.format.write_array_header_2_0(meta_out, {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False, "shape": meta_shape})
            writer = csv.writer(index_out)
            writer.writerow(("split", "object_id", "alert", "position"))
            for shard_id in sorted(map(int, final["shards"])):
                sr = source / f"shard_{shard_id:03d}"
                print(f"[fast-cache] reading shard {shard_id + 1}/{plan['n_shards']} "
                      f"rows_done={written}/{count}", flush=True)
                try:
                    info = json.loads((sr / "done.json").read_text())
                    if info.get("plan_sha256") != final["plan_sha256"]:
                        raise ValueError(f"Wrong shard plan {shard_id}")
                    if sha256(sr / "stage_index.csv") != info["index_sha256"]:
                        raise ValueError(f"Modified stage index {shard_id}")
                    n = int(info["stages"])
                    # Read the complete shard through buffered file I/O.
                    # A normal np.load() returns a resident ndarray, not mmap.
                    with (sr / "images.npy").open("rb") as inp:
                        im = np.load(inp, allow_pickle=False)
                    with (sr / "metadata.npy").open("rb") as inp:
                        md = np.load(inp, allow_pickle=False)
                    if im.shape != (n, 3, 63, 63) or md.shape != (n, 55):
                        raise ValueError(f"Invalid shape in shard {shard_id}")
                    if im.dtype != dtype or md.dtype != dtype:
                        raise ValueError(f"Wrong tensor dtype in shard {shard_id}")
                    with (sr / "stage_index.csv").open(newline="", encoding="utf-8") as sf:
                        stage_rows = list(csv.DictReader(sf))
                    if len(stage_rows) != n or written + n > count:
                        raise ValueError(f"Invalid number of stages in shard {shard_id}")
                    positions = [int(r["position"]) for r in stage_rows]
                    if sorted(positions) != list(range(n)):
                        raise ValueError(f"Non-bijective shard positions in {shard_id}")
                    for r in stage_rows:
                        split, oid, alert = r["split"], r["object_id"], int(r["alert"])
                        key = (split, oid, alert)
                        if key in seen or split not in per_split:
                            raise ValueError(f"Duplicated or invalid stage: {key}")
                        seen.add(key)
                        per_split[split] += 1
                        writer.writerow((split, oid, alert, written + int(r["position"])))
                    for offset in range(0, n, batch_rows):
                        end = min(n, offset + batch_rows)
                        # Buffered file writes; do not call ndarray.tofile(),
                        # which bypasses Python's write buffering.
                        images_out.write(memoryview(np.ascontiguousarray(im[offset:end])).cast('B'))
                        meta_out.write(memoryview(np.ascontiguousarray(md[offset:end])).cast('B'))
                    written += n
                    # Each shard flushes buffers, making I/O failures observable
                    # with shard context. fsync at the end precedes publication.
                    images_out.flush(); meta_out.flush(); index_out.flush()
                except (OSError, ValueError) as err:
                    raise RuntimeError(f"Failed consolidating source shard {shard_id:03d} "
                                       f"from {sr}: {err}") from err
                finally:
                    # The next iteration should not retain the prior shard.
                    if 'im' in locals():
                        del im
                    if 'md' in locals():
                        del md
                print(f"[fast-cache] completed shard {shard_id + 1}/{plan['n_shards']} "
                      f"stages={written}/{count} seconds={time.perf_counter()-started:.1f}",
                      flush=True)
            if written != count or per_split != final["manifest_rows"]:
                raise ValueError(f"Incomplete stage copy: {written}, {per_split}")
            for fileobj in (images_out, meta_out, index_out):
                fileobj.flush()
                os.fsync(fileobj.fileno())
        # Check exact NPY file lengths before reading or publication.
        # The files have already been closed and synced: the size here is
        # informative; validate() performs the actual NPY shape checks.
        manifest = {
            "format_version": FORMAT,
            "source": SOURCE,
            "source_cache": str(source.resolve()),
            "source_cache_manifest_sha256": sha256(source / "cache_manifest.json"),
            "source_plan_sha256": final["plan_sha256"],
            "manifest_sha256": final["manifest_sha256"],
            "preprocessing_identity": final["preprocessing_identity"],
            "labels_saved_in_cache": False,
            "test_split_accessed": False,
            "unique_objects": final["unique_objects"],
            "unique_stages": count,
            "manifest_rows": per_split,
            "stage_index_sha256": sha256(tmp / "stage_index.csv"),
            "images_nbytes": (tmp / "images.npy").stat().st_size,
            "metadata_nbytes": (tmp / "metadata.npy").stat().st_size,
            "build_seconds": time.perf_counter() - started,
            "writer": "buffered_npy_sequential_v2",
        }
        write_json_atomic(tmp / "fast_cache_manifest.json", manifest)
        # Validate before publishing the completed directory.
        checks = validate(source=source, target=tmp, samples_per_shard=3)
        print(f"[fast-cache] verified prepublication: {checks}", flush=True)
        os.rename(tmp, target)
        print(f"[fast-cache] published at {target}", flush=True)
        return manifest
    except BaseException:
        print(f"[fast-cache] build incomplete; partial files preserved at {tmp}", flush=True)
        raise


def validate(*, source: Path, target: Path, samples_per_shard: int = 3) -> dict:
    source, target = Path(source), Path(target)
    original, plan = read_identity(source)
    manifest = json.loads((target / "fast_cache_manifest.json").read_text())
    if manifest["source_cache_manifest_sha256"] != sha256(source / "cache_manifest.json"):
        raise ValueError("Source cache changed after consolidation")
    if manifest["source_plan_sha256"] != original["plan_sha256"]:
        raise ValueError("Source plan changed")
    if sha256(target / "stage_index.csv") != manifest["stage_index_sha256"]:
        raise ValueError("Consolidated stage index changed")
    images = np.load(target / "images.npy", allow_pickle=False)
    md = np.load(target / "metadata.npy", allow_pickle=False)
    n = original["unique_stages"]
    if images.shape != (n, 3, 63, 63) or md.shape != (n, 55):
        raise ValueError("Consolidated arrays have incorrect shapes")
    if images.dtype != np.float32 or md.dtype != np.float32:
        raise ValueError("Consolidated dtype mismatch")
    seen = set()
    index = {}
    counts = {"train": 0, "validation": 0}
    with (target / "stage_index.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r["split"], r["object_id"], int(r["alert"])
            pos = int(r["position"])
            if key in seen or not 0 <= pos < n or key[0] not in counts:
                raise ValueError("Invalid consolidated stage key or row position")
            seen.add(key)
            index[key] = pos
            counts[key[0]] += 1
    if len(seen) != n or counts != original["manifest_rows"] or len(set(index.values())) != n:
        raise ValueError("Consolidated index does not bijectively cover the source stages")
    n_checks = 0
    for shard_id in sorted(map(int, original["shards"])):
        sr = source / f"shard_{shard_id:03d}"
        sim = np.load(sr / "images.npy", allow_pickle=False)
        smd = np.load(sr / "metadata.npy", allow_pickle=False)
        with (sr / "stage_index.csv").open(newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        selected = np.linspace(0, len(records)-1, min(len(records), samples_per_shard), dtype=int)
        for j in selected:
            r = records[int(j)]
            old = int(r["position"])
            new = index[(r["split"], r["object_id"], int(r["alert"]))]
            if not np.array_equal(images[new], sim[old]) or not np.array_equal(md[new], smd[old]):
                raise ValueError(f"Numerical mismatch: shard={shard_id} row={old}")
            n_checks += 1
    return {"verified": True, "split_rows": counts, "unique_stages": len(seen),
            "matched_original_shard_samples": n_checks,
            "nvme_bytes": (target / "images.npy").stat().st_size +
                         (target / "metadata.npy").stat().st_size}


class FastHistoricalDataset(Dataset):
    """Manifest-based indices into NVMe-mapped immutable full-history tensors.

    No eager copying, no label information in stage index. To form AL L_t/U_t,
    pass a subset of the original historical rows and optionally omit label_col.
    """
    def __init__(self, *, rows: list[dict], cache_root: Path, split: str,
                 object_column: str, label_column: str | None):
        root = Path(cache_root)
        self.manifest = json.loads((root / "fast_cache_manifest.json").read_text())
        if self.manifest.get("format_version") != FORMAT or self.manifest.get("source") != SOURCE:
            raise ValueError("Invalid full-history fast-cache identity")
        if split not in ("train", "validation"):
            raise ValueError("Unsupported split")
        self.images = np.load(root / "images.npy", mmap_mode="r")
        self.metadata = np.load(root / "metadata.npy", mmap_mode="r")
        lookup = {}
        with (root / "stage_index.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    key = r["object_id"], int(r["alert"])
                    if key in lookup:
                        raise ValueError("Duplicate stage in consolidated cache")
                    lookup[key] = int(r["position"])
        self.positions = []
        for r in rows:
            key = str(r[object_column]), int(r["alert"])
            if key not in lookup:
                raise KeyError(f"Uncached historical stage for {split}: {key}")
            self.positions.append(lookup[key])
            if label_column is not None:
                label_index(r[label_column])
        self.rows = rows
        self.object_column = object_column
        self.label_column = label_column
        self.split = split

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        pos = self.positions[i]
        item = {
            "obj_id": str(r[self.object_column]),
            "alert": int(r["alert"]),
            "image": torch.from_numpy(np.asarray(self.images[pos]).copy()),
            "metadata": torch.from_numpy(np.asarray(self.metadata[pos]).copy()),
        }
        if self.label_column is not None:
            item["target_index"] = label_index(r[self.label_column])
        return item
