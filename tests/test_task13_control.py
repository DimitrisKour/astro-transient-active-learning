"""No-GPU, no-Slurm tests of Task13's deployment control plane."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/adaptive"))
import task13  # noqa: E402


class TestTask13Control(unittest.TestCase):
    def test_filesystem_probe_writes_fsyncs_and_removes_only_its_temp_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(task13, "DATA", Path(folder)), patch.object(task13, "verify"):
                result = task13.filesystem_probe(job_dir=folder, size_bytes=1024)
            self.assertTrue(result["passed"])
            self.assertEqual(result["bytes"], 1024)
            probe_dir = Path(folder) / "runs" / task13.SUITE / "shared/preflight"
            self.assertEqual(list(probe_dir.iterdir()), [])

    def test_24_fits_and_four_cycles_are_pinned(self):
        plan = task13.scientific_plan()
        self.assertEqual(plan["training_jobs"], 24)
        self.assertEqual(plan["acquisition_cycles"], 4)
        self.assertEqual(len(tuple(task13.planned_jobs())), 24)
        args = task13.runner_args(135, "tailor_diversity")
        self.assertEqual(args[args.index("--max-cycles") + 1], "4")
        self.assertNotIn("--complete-pool", args)
        self.assertNotIn("--extend-budget", args)

    def test_submit_keeps_every_job_held_and_dependency_ordered(self):
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)
            seen = []

            def fake_sbatch(cmd, text):
                self.assertTrue(text)
                seen.append(cmd)
                return f"{500000 + len(seen)}\n"

            with patch.object(task13, "verify"), patch.object(task13, "check_pilots"), patch.object(
                task13.subprocess, "check_output", side_effect=fake_sbatch
            ):
                task13.submit(job_dir=job)
            ledger = json.loads((job / "submitted.json").read_text())
            self.assertEqual(ledger["status"], "held_awaiting_explicit_release")
            self.assertEqual(len(ledger["jobs"]), 24)
            self.assertEqual(len(seen), 24)
            self.assertTrue(all("--hold" in cmd for cmd in seen))
            self.assertTrue(all("scontrol" not in cmd for cmd in seen))
            for entry, command in zip(ledger["jobs"], seen, strict=True):
                if entry["cycle"] == 1:
                    self.assertIsNone(entry["parent"])
                    self.assertNotIn("--dependency", command)
                else:
                    self.assertIn("--dependency", command)
                    self.assertIn("afterok:" + entry["parent"], command)

            releases = []

            def fake_release(cmd, check):
                self.assertEqual(cmd[:2], ["scontrol", "release"])
                releases.append(cmd[2])

            with patch.object(task13, "verify"), patch.object(task13, "check_pilots"), patch.object(
                task13.subprocess, "check_output", return_value="JobState=PENDING Reason=JobHeldUser\n"
            ), patch.object(task13.subprocess, "run", side_effect=fake_release):
                task13.release(job_dir=job)
            self.assertEqual(len(releases), 24)
            self.assertEqual(json.loads((job / "submitted.json").read_text())["status"], "released")

    def test_never_overwrite_submitted_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)
            (job / "submitted.json").write_text('{"jobs": []}\n')
            with patch.object(task13, "verify"), patch.object(task13, "check_pilots"):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    task13.submit(job_dir=job)


if __name__ == "__main__":
    unittest.main()
