"""Synthetic acquisition and isolated-archive checks; no real API requests."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from build_mean_ble_zip import build_zip
from streaming_ingestion import AcquisitionHook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from empirical_mean_ble_acquisition import labeling, model


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.target = [
            {"normalized_name": "synthetic-system", "harness": "synthetic-harness"},
            {"item_content": "Synthetic target", "benchmark_id": "synthetic-benchmark"},
        ]
        self.labeled = [[copy.deepcopy(self.target), 0]]
        self.context = {"labels_remaining": 5, "items_remaining": 20,
                        "labels_acquired": 1, "max_labels": 31}
        self.hook = AcquisitionHook(labeling.acquisition_function)

    def test_streaming_hook_uses_ble_not_the_empirical_mean(self):
        self.assertEqual(self.hook.mode, "stream")
        self.assertFalse(self.hook.needs_prediction)
        before = copy.deepcopy([self.target, self.labeled, self.context])
        self.assertEqual(model.predict(self.target, self.labeled), 0.0)
        with patch.object(labeling, "_ble_prediction", return_value=0.5) as ble:
            self.assertTrue(self.hook(self.target, 0.0, self.labeled, self.context))
            ble.assert_called_once_with(self.target, self.labeled)
        with patch.object(labeling, "_ble_prediction", return_value=0.9):
            self.assertFalse(self.hook(self.target, 0.5, self.labeled, self.context))
        self.assertEqual([self.target, self.labeled, self.context], before)

    def test_budget_shortcuts_and_mean_predictions_do_not_call_ble(self):
        with patch.object(labeling, "_ble_prediction", side_effect=AssertionError("unexpected BLE call")):
            self.assertFalse(self.hook(self.target, labeled=self.labeled,
                                       context=dict(self.context, labels_remaining=0)))
            self.assertTrue(self.hook(self.target, labeled=self.labeled,
                                      context=dict(self.context, items_remaining=5)))
            self.assertEqual(model.predict(self.target, self.labeled), 0.0)
            self.assertEqual(model.predict(self.target, []), 0.5)

    def test_ble_errors_propagate_and_streaming_context_is_required(self):
        with patch.object(labeling, "_ble_prediction", side_effect=RuntimeError("synthetic outage")):
            with self.assertRaisesRegex(RuntimeError, "synthetic outage"):
                self.hook(self.target, labeled=self.labeled, context=self.context)
        with self.assertRaisesRegex(ValueError, "streaming context"):
            labeling.acquisition_function(self.target)


class PackagedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ("empirical_mean_ble_acquisition", "empirical_mean"):
            shutil.copytree(ROOT / directory, self.root / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        ble = self.root / "ble"
        ble.mkdir()
        for name in ("model.py", "labeling.py", "requirements.txt", "submission_config.example.json"):
            shutil.copyfile(ROOT / "ble" / name, ble / name)
        for name in ("src", "comp_pipeline"):
            shutil.copytree(ROOT / "ble" / name, ble / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        self.private_marker = "synthetic-local-credential"
        (ble / "submission_config.json").write_text(json.dumps({
            "api_keys": {"OPENAI_API_KEY": self.private_marker},
        }))
        data = self.root / "payload/data"
        data.mkdir(parents=True)
        (data / "corpus_manifest.json").write_text(json.dumps({
            "corpus_scope": "paec-public-training",
            "db_repo": "aims-foundations/measurement-db",
            "revision": "a" * 40,
            "files": [],
        }))
        (data / "unlisted-private-fixture.txt").write_text("not part of public payload")

    def test_public_bundle_is_self_contained_and_evaluation_never_loads_ble(self):
        path = build_zip(self.root, public=True)
        extracted = self.root / "extracted"
        with ZipFile(path) as archive:
            config = json.loads(archive.read("ble/submission_config.json"))
            self.assertFalse(any(config["api_keys"].values()))
            self.assertIn("ble/data/corpus_manifest.json", archive.namelist())
            self.assertIn("requirements.txt", archive.namelist())
            self.assertNotIn("ble/data/unlisted-private-fixture.txt", archive.namelist())
            for name in archive.namelist():
                self.assertNotIn(self.private_marker.encode(), archive.read(name))
            archive.extractall(extracted)
        script = r'''
import importlib.util, json, pathlib, socket, sys
from unittest.mock import patch
root = pathlib.Path(sys.argv[1])
# Like the validator, remove the temporary import path after loading entry points.
sys.path.insert(0, str(root))
import model, labeling
sys.path.remove(str(root))
subject = {"normalized_name": "synthetic-system", "harness": "synthetic-harness"}
item = {"item_content": "Synthetic target", "benchmark_id": "synthetic-benchmark"}
target = [subject, item]
labels = [[[subject, dict(item, item_content="Synthetic acquired item")], 0]]
context = {"labels_remaining": 3, "items_remaining": 10}
with patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")):
    for evidence, expected in [(labels, 0.0), ([], 0.5), (labels, 0.0)]:
        assert model.predict(target, evidence) == expected
    assert "ble.model" not in sys.modules
    # Exercise the actual nested BLE module with a scripted agent result.
    from ble import model as ble
    from comp_pipeline.script.dataroot import data_root
    assert pathlib.Path(ble.__file__) == root / "ble/model.py"
    assert pathlib.Path(data_root()) == root / "ble/data"
    with patch.object(ble, "run_agent", return_value={"submitted": True, "forecast": 0.5}) as agent:
        assert labeling.acquisition_function(target, labeled=labels, context=context)
        question, config = agent.call_args.args
        assert "[INCORRECT] Synthetic acquired item" in question["background"]
        assert config.reasoning_effort == "medium"
        assert config.max_steps == 5
        before = agent.call_count
        assert model.predict(target, labels) == 0.0
        assert agent.call_count == before
    with patch.object(ble, "run_agent", return_value={"submitted": True, "forecast": 0.9}):
        assert not labeling.acquisition_function(target, labeled=labels, context=context)
print("Isolated archive: mean-only evaluation, nested BLE acquisition, and public payload passed")
'''
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("BLF_", "OPENAI_", "ANTHROPIC_"))}
        result = subprocess.run([sys.executable, "-I", "-c", script, str(extracted)],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_private_build_is_explicit_and_hf_declaration_stays_at_root(self):
        public = build_zip(self.root, public=True, mode="hf", repo_id="example/public-payload")
        with ZipFile(public) as archive:
            self.assertIn("models.txt", archive.namelist())
            self.assertNotIn("ble/models.txt", archive.namelist())
            self.assertEqual(json.loads(archive.read("ble/submission_config.json"))["hf_data_repo"],
                             "example/public-payload")
        private = build_zip(self.root, public=False)
        self.assertNotEqual(public, private)
        with ZipFile(private) as archive:
            self.assertEqual(json.loads(archive.read("ble/submission_config.json"))["api_keys"]["OPENAI_API_KEY"],
                             self.private_marker)

    def test_static_validation_never_executes_submission_code(self):
        path = build_zip(self.root, public=True)
        # A module-level sentinel would fail immediately if static validation imported it.
        with ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        files["model.py"] = b"raise RuntimeError('participant code must not execute')\n"
        with ZipFile(path, "w") as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
        result = subprocess.run([sys.executable, str(ROOT / "check_submission_zip.py"),
                                 "--static-only", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Static checks only", result.stdout)


if __name__ == "__main__":
    unittest.main()
