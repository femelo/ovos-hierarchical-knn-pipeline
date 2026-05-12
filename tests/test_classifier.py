"""Unit tests for HierarchicalPairKNNClassifier.

These tests build a tiny synthetic index from pre-computed embeddings (so no
encoder model is downloaded) and exercise the build / from_disk / inference
paths end-to-end.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier


def _make_synthetic_data(rng_seed: int = 0):
    """Return (labels, embeddings) for a tiny 2-level hierarchy.

    Hierarchy:
        domain_a:intent_x  (5 docs)
        domain_a:intent_y  (5 docs)
        domain_b:intent_z  (5 docs)
        domain_b:intent_w  (5 docs)

    Embeddings are 8-dim, with each class clustered around a deterministic
    direction so KNN can actually separate them.
    """
    rng = np.random.default_rng(rng_seed)
    classes = [
        "domain_a:intent_x",
        "domain_a:intent_y",
        "domain_b:intent_z",
        "domain_b:intent_w",
    ]
    centers = {
        "domain_a:intent_x": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "domain_a:intent_y": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "domain_b:intent_z": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        "domain_b:intent_w": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    }
    labels = []
    embeddings = []
    for c in classes:
        for _ in range(5):
            v = centers[c] + 0.05 * rng.standard_normal(8).astype(np.float32)
            embeddings.append(v)
            labels.append(c)
    embeddings = np.stack(embeddings)
    return classes, labels, embeddings


class TestClassifierHelpers(unittest.TestCase):
    """Functions that don't need an index or encoder."""

    def setUp(self):
        self.clf = HierarchicalPairKNNClassifier(
            classes=["a:x", "a:y", "b:z"], k=3, nlist=2, pq_m=2,
        )

    def test_init_sets_attributes(self):
        self.assertEqual(self.clf.k, 3)
        self.assertEqual(self.clf.sep, ":")
        self.assertIsNone(self.clf.index)
        self.assertEqual(len(self.clf.classes), 2)  # 2-level hierarchy

    def test_get_depth(self):
        self.assertEqual(self.clf.get_depth(), 2)

    def test_get_classes_levels(self):
        levels = self.clf._get_classes_levels(["a:x", "a:y", "b:z"])
        self.assertEqual(len(levels), 2)
        self.assertIn("a", levels[0])
        self.assertIn("b", levels[0])

    def test_get_classes(self):
        levels = self.clf._get_classes_levels(["a:x", "a:y", "b:z"], unique=False)
        l1 = self.clf._get_classes(levels, level=1)
        self.assertEqual(set(l1), {"a", "b"})
        l2 = self.clf._get_classes(levels, level=2)
        self.assertEqual(set(l2), {"a:x", "a:y", "b:z"})

    def test_get_previous_level_top(self):
        # Level 1 has no previous — must return None
        result = self.clf._get_previous_level("a", self.clf.classes, level=1)
        self.assertIsNone(result)

    def test_get_previous_level_traces_parent(self):
        # subclass "a:x" at level 2 should map back to "a" at level 1
        result = self.clf._get_previous_level("a:x", self.clf.classes, level=2)
        self.assertEqual(result, "a")


class TestProbabilitySolvers(unittest.TestCase):
    """The Wu-Lin pairwise solvers operate on plain numpy arrays."""

    def setUp(self):
        self.clf = HierarchicalPairKNNClassifier(
            classes=["a:x", "a:y"], k=3, gamma=1.0, tau=0.05, margin=0.10,
        )

    def test_basic_probabilities_sum_to_one(self):
        distances = np.array([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float32)
        labels = np.array(["a", "b", "a", "b", "a"])
        classes = np.array(["a", "b"])
        probs = self.clf._get_probabilities(distances, classes, labels)
        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)
        self.assertTrue((probs >= 0).all())

    def test_adaptive_neighborhood_exact_match_short_circuits(self):
        # When the nearest distance is below `tau`, the winner takes all.
        distances = np.array([0.01, 0.5, 0.6], dtype=np.float32)
        labels = np.array(["a", "b", "b"])
        classes = np.array(["a", "b"])
        probs = self.clf._get_probabilities_with_adaptive_neighborhood(
            distances, classes, labels,
        )
        self.assertAlmostEqual(probs[0], 1.0)
        self.assertAlmostEqual(probs[1], 0.0)

    def test_adaptive_neighborhood_distributes(self):
        distances = np.array([0.2, 0.21, 0.22, 0.23], dtype=np.float32)
        labels = np.array(["a", "b", "a", "b"])
        classes = np.array(["a", "b"])
        probs = self.clf._get_probabilities_with_adaptive_neighborhood(
            distances, classes, labels, d_anchor=0.2,
        )
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)
        self.assertTrue((probs >= 0).all())


class TestBuildAndPredict(unittest.TestCase):
    """End-to-end: build a small index with synthetic embeddings, then load
    and predict against it.  Uses a flat index (n < nlist*256)."""

    @classmethod
    def setUpClass(cls):
        cls.classes, cls.labels, cls.embeddings = _make_synthetic_data()
        cls.tmp = tempfile.mkdtemp(prefix="knn_test_")
        cls.index_dir = os.path.join(cls.tmp, "idx")

        builder = HierarchicalPairKNNClassifier(
            classes=cls.classes, k=3, n=2, nlist=2, pq_m=2,
            gamma=1.0, tau=0.05, margin=0.10,
            model_path=".",  # relative path resolved against index_dir on load
        )
        # Stub the encoder so build()'s metadata write does not download a model.
        class _FakeEncoder:
            onnx_filename = None
        builder._encoder = _FakeEncoder()
        builder.build(
            documents=None, labels=cls.labels, index_dir=cls.index_dir,
            embeddings=cls.embeddings.copy(),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self):
        return HierarchicalPairKNNClassifier.from_disk(self.index_dir)

    def test_build_writes_artifacts(self):
        for name in ("index.faiss", "label_ids.npy", "class_names.npy",
                     "class_to_train_ids.pkl", "meta.pkl"):
            self.assertTrue(
                os.path.exists(os.path.join(self.index_dir, name)),
                f"missing {name}",
            )

    def test_from_disk_round_trip(self):
        clf = self._load()
        self.assertEqual(clf.k, 3)
        self.assertEqual(clf.sep, ":")
        self.assertIsNotNone(clf.index)
        self.assertEqual(clf.get_depth(), 2)

    def test_predict_proba_top_class_matches_input_cluster(self):
        clf = self._load()
        # Query close to the domain_a:intent_x cluster center
        query = np.array(
            [[1.0, 0.02, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        probs_list = clf.predict_proba(query)
        self.assertEqual(len(probs_list), 1)
        probs = probs_list[0]
        # Highest-probability label should be domain_a:intent_x
        top = max(probs.items(), key=lambda kv: kv[1])
        self.assertEqual(top[0], "domain_a:intent_x")

    def test_predict_returns_string_labels(self):
        clf = self._load()
        query = np.array(
            [[0, 0, 1.0, 0.02, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        result = clf.predict(query)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "domain_b:intent_z")

    def test_predict_proba_dimension_mismatch_raises(self):
        clf = self._load()
        wrong = np.zeros((1, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            clf.predict_proba(wrong)

    def test_predict_proba_level_one_only(self):
        clf = self._load()
        query = np.array(
            [[1.0, 0.02, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        probs = clf.predict_proba(query, level=1)[0]
        # At level=1 we only get domain-level labels.
        self.assertTrue(any(k in {"domain_a", "domain_b"} for k in probs))


class TestActiveDomains(unittest.TestCase):
    """`set_active_domains` constrains L1 search to a subset of domains."""

    @classmethod
    def setUpClass(cls):
        cls.classes, cls.labels, cls.embeddings = _make_synthetic_data(rng_seed=1)
        cls.tmp = tempfile.mkdtemp(prefix="knn_dom_")
        cls.index_dir = os.path.join(cls.tmp, "idx")
        builder = HierarchicalPairKNNClassifier(
            classes=cls.classes, k=3, n=2, nlist=2, pq_m=2,
            model_path=".",
        )
        class _FakeEncoder:
            onnx_filename = None
        builder._encoder = _FakeEncoder()
        builder.build(
            documents=None, labels=cls.labels, index_dir=cls.index_dir,
            embeddings=cls.embeddings.copy(),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_set_active_domains_filters_results(self):
        clf = HierarchicalPairKNNClassifier.from_disk(self.index_dir)
        clf.set_active_domains(["domain_a"])
        query = np.array(
            [[0, 0, 1.0, 0.02, 0, 0, 0, 0]],  # nearest to domain_b
            dtype=np.float32,
        )
        probs = clf.predict_proba(query)[0]
        # With only domain_a active, the prediction must come from domain_a.
        top = max(probs.items(), key=lambda kv: kv[1])
        self.assertTrue(top[0].startswith("domain_a:"),
                        f"expected domain_a prediction, got {top[0]}")

    def test_set_active_domains_none_clears_filter(self):
        clf = HierarchicalPairKNNClassifier.from_disk(self.index_dir)
        clf.set_active_domains(["domain_a"])
        clf.set_active_domains(None)
        self.assertIsNone(clf._domain_bitmap)
        self.assertIsNone(clf._active_domain_mask)
        self.assertIsNone(clf._domain_train_ids)

    def test_set_active_domains_empty_list_clears_filter(self):
        clf = HierarchicalPairKNNClassifier.from_disk(self.index_dir)
        clf.set_active_domains([])
        self.assertIsNone(clf._domain_bitmap)

    def test_set_active_domains_unknown_domain_clears_filter(self):
        clf = HierarchicalPairKNNClassifier.from_disk(self.index_dir)
        clf.set_active_domains(["nonexistent_domain"])
        # No domains match → filter is cleared rather than producing an empty
        # search pool.
        self.assertIsNone(clf._domain_bitmap)


if __name__ == "__main__":
    unittest.main()
