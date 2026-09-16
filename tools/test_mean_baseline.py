"""Check group isolation and independent checkpoint calls for the mean baseline."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from build_mean_zip import build_zip


def load_predict(path):
    spec = importlib.util.spec_from_file_location("mean_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict


class MeanBaselineTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.predict = load_predict(root / "empirical_mean/model.py")
        self.subject = {"normalized_name": "synthetic-model", "harness": "harness-a"}
        self.item = {"item_content": "Synthetic target", "benchmark_id": "benchmark_274926"}
        self.target = [self.subject, self.item]

    def observation(self, label, *, subject=None, benchmark=None):
        item = dict(self.item, item_content="Synthetic acquired item")
        if benchmark is not None:
            item["benchmark_id"] = benchmark
        return [[dict(self.subject if subject is None else subject), item], label]

    def test_other_benchmarks_and_subject_settings_do_not_change_prediction(self):
        labels = [self.observation(y) for y in (1, 0, 0)]
        unrelated = [self.observation(1, benchmark="benchmark_582731") for _ in range(31)]
        unrelated += [self.observation(1, subject=dict(self.subject, harness="harness-b"))]
        before = copy.deepcopy([self.target, labels, unrelated])
        self.assertAlmostEqual(self.predict(self.target, labels + unrelated), 1 / 3)
        self.assertEqual(self.predict(self.target, unrelated), 0.5)
        self.assertEqual([self.target, labels, unrelated], before)

    def test_checkpoints_can_run_out_of_order_without_retaining_evidence(self):
        labels = [self.observation(y) for y in [1] + [0] * 30]
        for n in (31, 0, 7, 1, 15, 3, 0):
            with self.subTest(budget=n):
                expected = 1 / n if n else 0.5
                self.assertEqual(self.predict(self.target, labels[:n]), expected)
        self.assertEqual(self.predict(self.target, [self.observation(0)]), 0.0)

    def test_missing_group_metadata_does_not_silently_pool_benchmarks(self):
        labels = [self.observation(1)]
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            self.predict([self.subject, {"item_content": "target"}], labels)
        del labels[0][0][1]["benchmark_id"]
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            self.predict(self.target, labels)
        with self.assertRaisesRegex(ValueError, "binary"):
            self.predict(self.target, [self.observation(float("nan"))])

    def test_zip_imports_independently_with_no_payload_or_acquisition_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_zip(Path(directory) / "mean.zip")
            first = path.read_bytes()
            self.assertEqual(build_zip(path).read_bytes(), first)
            with ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {"model.py", "README.md"})
                archive.extractall(Path(directory) / "extracted")
            predict = load_predict(Path(directory) / "extracted/model.py")
            self.assertEqual(predict(self.target), 0.5)
            self.assertEqual(predict(self.target, [self.observation(0)]), 0.0)


if __name__ == "__main__":
    unittest.main()
