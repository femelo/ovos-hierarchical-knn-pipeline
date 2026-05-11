"""
HierarchicalPairKNNClassifier backed by a FAISS IVF+PQ index.

Two-phase usage
---------------
Offline (powerful machine):
    clf = HierarchicalPairKNNClassifier(classes, ...)
    clf.build(documents, labels, index_dir="path/to/save")

On-device (Raspberry Pi / edge):
    clf = HierarchicalPairKNNClassifier.from_disk("path/to/save")
    clf.predict(["what time is it?"])
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import Any
from tqdm import tqdm
from .encoders import AnyEncoder, load_encoder


class HierarchicalPairKNNClassifier:
    def __init__(
        self,
        classes: list,
        k: int = 10,
        n: int = 5,
        sep: str = ":",
        nlist: int = 1024,
        pq_m: int = 16,
        nprobe: int = 32,
        model_path: str = "m2v-labse",
        gamma: float = 1.0,
        tau: float = 0.05,
        margin: float = 0.10,
        anchor_to_global: bool = True,
        encoder_file: str | None = None,
    ) -> None:
        self.k = k
        self.n = n
        self.sep = sep
        self.nlist = nlist
        self.pq_m = pq_m
        self.nprobe = nprobe

        self.tau = tau
        self.margin = margin
        self.anchor_to_global = anchor_to_global

        self.model_path = model_path
        self.encoder_file = encoder_file  # ONNX filename override; not persisted in meta
        self.gamma = gamma
        self.classes_flattened = np.unique(classes)
        self.classes = self._get_classes_levels(classes)

        self.index: faiss.Index | None = None
        self.level_label_ids: list[np.ndarray] = []
        self.level_class_names: list[np.ndarray] = []
        self.class_to_train_ids: list[list[np.ndarray]] = []
        self.min_domain_count: int = 0
        self.probabilities: list = []
        self.selected_classes: list = []

        self._encoder: AnyEncoder | None = None

        # Set by set_active_domains(); None means search the full index.
        self._domain_bitmap: np.ndarray | None = None
        self._active_domain_mask: np.ndarray | None = None
        self._domain_train_ids: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_classes_levels(
        self, labels: list[str], unique: bool = True
    ) -> list[tuple[str, ...]]:
        if unique:
            labels = np.unique(labels).tolist()
        return [*zip(*map(lambda c: c.split(self.sep), labels))]

    def _get_classes(self, labels: list, level: int, unique: bool = False) -> list[Any]:
        level = min(level, len(labels))
        classes = list(map(lambda t: self.sep.join(list(t)), zip(*labels[:level])))
        if unique:
            classes = np.unique(classes).tolist()
        return classes

    def get_depth(self) -> int:
        return len(self.classes)

    @property
    def encoder(self) -> AnyEncoder:
        if self._encoder is None:
            self._encoder = load_encoder(self.model_path, onnx_filename=self.encoder_file)
        return self._encoder

    def _encode_documents(self, documents, show_progress_bar: bool = False) -> np.ndarray:
        return self.encoder.encode_documents(list(documents), show_progress_bar=show_progress_bar)

    def _encode_queries(self, documents, show_progress_bar: bool = False) -> np.ndarray:
        return self.encoder.encode_queries(list(documents), show_progress_bar=show_progress_bar)

    def _get_previous_level(
        self, subclass: str, labels: list, level: int | None = None
    ) -> str | None:
        if level is None:
            level = 0
            for i, subclasses in enumerate(labels):
                if subclass in subclasses:
                    level = i + 1
                    break
        level = min(level, len(labels))
        if level <= 1:
            return None
        prev = self._get_classes(labels, level=level - 1)
        curr = self._get_classes(labels, level=level)
        return prev[curr.index(subclass)]

    def _get_probabilities_with_adaptive_neighborhood(
        self,
        distances: np.ndarray,
        classes: np.ndarray,
        labels: np.ndarray,
        d_anchor: float | None = None,
        epsilon: float = 0.001,
    ) -> np.ndarray:
        """Vectorized Wu-Lin pairwise probability estimation with distance decay,
        Exact Match Override, and Adaptive Margin limits.

        d_anchor: nearest-neighbor distance from the unfiltered global search.
        When the candidate pool is restricted to a domain subset, passing the
        true nearest-neighbor distance here keeps the shell boundary and decay
        anchored to the full-space geometry, so margin/gamma retain their
        original meaning regardless of how sparse the filtered pool is.
        If None, falls back to the nearest distance in the candidate pool.
        """
        tau = self.tau
        margin = self.margin

        n_classes = len(classes)
        n_samples = len(distances)

        sort_idx = np.argsort(distances)
        sorted_labels = labels[sort_idx]
        sorted_distances = distances[sort_idx]

        # Exact match uses the actual nearest candidate, not the global anchor —
        # it answers "is there a hit-point match in my candidate pool".
        if sorted_distances[0] <= tau:
            probs = np.zeros(n_classes)
            winner_idx = np.where(classes == sorted_labels[0])[0][0]
            probs[winner_idx] = 1.0
            return probs

        # Shell boundary and decay are anchored to the global (unfiltered) nearest
        # neighbor.  This keeps margin/gamma invariant when the candidate pool is a
        # domain-restricted subset of the full index.
        ref_dist = d_anchor if d_anchor is not None else sorted_distances[0]

        if self.gamma > 0:
            sigma = margin / self.gamma
            centered = sorted_distances - ref_dist
            decay = np.exp(-(centered ** 2) / (2 * (sigma ** 2)))
            decay = np.clip(decay, 1e-10, 1.0)
        else:
            decay = np.ones_like(sorted_distances)

        valid_mask = sorted_distances <= (ref_dist + margin)
        decay = decay * valid_mask

        class_weights = (sorted_labels[None, :] == classes[:, None]) * decay[None, :]
        cum_weights = np.cumsum(class_weights, axis=1)
        combined_cum = cum_weights[:, None, :] + cum_weights[None, :, :]

        total_valid_mass = np.sum(decay)
        dynamic_k = min(self.k, total_valid_mass)
        dynamic_k = max(dynamic_k, 1e-5)

        reached = combined_cum >= dynamic_k
        first_t = np.where(reached.any(axis=2), np.argmax(reached, axis=2), n_samples - 1)
        r = cum_weights[np.arange(n_classes)[:, None], first_t] / dynamic_k

        is_zero = r == 0.0
        r += (is_zero.astype(float) - is_zero.T.astype(float)) * epsilon

        q = -(r * r.T)
        np.fill_diagonal(q, np.sum(r ** 2, axis=0))

        e = np.ones((n_classes, 1))
        A = np.block([[q, e], [e.T, np.zeros((1, 1))]])
        b = np.zeros(n_classes + 1)
        b[-1] = 1.0
        x = np.linalg.solve(A, b)
        probs = x[:-1]

        return probs / probs.sum()

    def _get_probabilities(
        self,
        distances: np.ndarray,
        classes: np.ndarray,
        labels: np.ndarray,
        epsilon: float = 0.001,
    ) -> np.ndarray:
        """Vectorized Wu-Lin pairwise probability estimation with exponential distance decay."""
        n_classes = len(classes)
        n_samples = len(distances)

        sort_idx = np.argsort(distances)
        sorted_labels = labels[sort_idx]
        sorted_distances = distances[sort_idx]

        decay = np.exp(-self.gamma * sorted_distances)

        class_weights = (sorted_labels[None, :] == classes[:, None]) * decay[None, :]
        cum_weights = np.cumsum(class_weights, axis=1)
        combined_cum = cum_weights[:, None, :] + cum_weights[None, :, :]

        reached = combined_cum >= self.k
        first_t = np.where(reached.any(axis=2), np.argmax(reached, axis=2), n_samples - 1)
        r = cum_weights[np.arange(n_classes)[:, None], first_t] / self.k

        is_zero = r == 0.0
        r += (is_zero.astype(float) - is_zero.T.astype(float)) * epsilon

        q = -(r * r.T)
        np.fill_diagonal(q, np.sum(r ** 2, axis=0))

        e = np.ones((n_classes, 1))
        A = np.block([[q, e], [e.T, np.zeros((1, 1))]])
        b = np.zeros(n_classes + 1)
        b[-1] = 1.0
        x = np.linalg.solve(A, b)
        probs = x[:-1]
        return probs / probs.sum()

    # ------------------------------------------------------------------
    # Offline build
    # ------------------------------------------------------------------

    def build(
        self,
        documents,
        labels: list,
        index_dir: str | Path,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Build a FAISS IVF+PQ index and persist everything to `index_dir`.

        Pass `embeddings` (shape: [n, dim], float32) to skip encoding — useful
        when embeddings are pre-computed from a parquet dataset.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        train_labels = self._get_classes_levels(labels, unique=False)

        if embeddings is not None:
            print("Using pre-computed embeddings…")
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
        else:
            print("Encoding training documents…")
            embeddings = self._encode_documents(documents, show_progress_bar=True)

        d = embeddings.shape[1]

        level_label_ids = []
        level_class_names = []
        for lvl in range(1, self.get_depth() + 1):
            class_strings = np.array(self._get_classes(train_labels, lvl))
            unique_classes = np.unique(class_strings)
            class_to_id = {c: i for i, c in enumerate(unique_classes)}
            label_ids = np.array([class_to_id[c] for c in class_strings], dtype=np.uint32)
            level_label_ids.append(label_ids)
            level_class_names.append(unique_classes)

        n = len(embeddings)
        min_for_ivfpq = self.nlist * 256
        ivf_index: faiss.IndexIVFPQ | None = None
        if n >= min_for_ivfpq:
            print(f"Building FAISS IVF+PQ index (nlist={self.nlist}, pq_m={self.pq_m})…")
            quantizer = faiss.IndexFlatIP(d)
            ivf_index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.pq_m, 8, faiss.METRIC_INNER_PRODUCT)
            n_train_sample = min(n, self.nlist * 40)
            rng = np.random.default_rng(42)
            train_sample = embeddings[rng.choice(n, n_train_sample, replace=False)]
            print(f"Training quantizer on {n_train_sample} samples…")
            ivf_index.train(train_sample)

        index: faiss.IndexFlatIP | faiss.IndexIVFPQ = (
            ivf_index if ivf_index is not None
            else faiss.IndexFlatIP(d)
        )
        if ivf_index is None:
            print(f"Dataset too small for IVF+PQ ({n} < {min_for_ivfpq}); using flat index.")

        print("Adding all vectors to the index…")
        batch = 100_000
        for start in tqdm(range(0, n, batch), desc="indexing"):
            index.add(embeddings[start : start + batch])

        if ivf_index is not None:
            ivf_index.nprobe = self.nprobe

        class_to_train_ids = []
        for label_ids_arr, class_names_arr in zip(level_label_ids, level_class_names):
            ctids = [
                np.where(label_ids_arr == c)[0].astype(np.uint32)
                for c in range(len(class_names_arr))
            ]
            class_to_train_ids.append(ctids)

        min_domain_count = int(min(len(ids) for ids in class_to_train_ids[0]))

        faiss.write_index(index, str(index_dir / "index.faiss"))
        np.save(index_dir / "label_ids.npy", np.stack(level_label_ids))
        np.save(index_dir / "class_names.npy", np.array(level_class_names, dtype=object))
        with open(index_dir / "class_to_train_ids.pkl", "wb") as f:
            pickle.dump(class_to_train_ids, f)
        meta = {
            "k": self.k,
            "n": self.n,
            "sep": self.sep,
            "nlist": self.nlist,
            "pq_m": self.pq_m,
            "nprobe": self.nprobe,
            "model_path": self.model_path,
            "gamma": self.gamma,
            "tau": self.tau,
            "margin": self.margin,
            "anchor_to_global": self.anchor_to_global,
            "classes": self.classes,
            "classes_flattened": self.classes_flattened,
            "min_domain_count": min_domain_count,
        }
        with open(index_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        index_mb = (index_dir / "index.faiss").stat().st_size / 1e6
        labels_mb = (index_dir / "label_ids.npy").stat().st_size / 1e6
        print(f"Saved to {index_dir}  (index {index_mb:.1f} MB, labels {labels_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # On-device load
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(cls, index_dir: str | Path) -> "HierarchicalPairKNNClassifier":
        """Load a previously built index."""
        index_dir = Path(index_dir)
        with open(index_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj = cls.__new__(cls)
        obj.k = meta["k"]
        obj.n = meta["n"]
        obj.sep = meta["sep"]
        obj.nlist = meta["nlist"]
        obj.pq_m = meta["pq_m"]
        obj.nprobe = meta["nprobe"]
        raw_model_path = meta["model_path"]
        # A relative model_path (e.g. ".") is resolved against index_dir so that
        # the encoder and index can be distributed together (e.g. from HuggingFace).
        if raw_model_path.startswith("."):
            obj.model_path = str((index_dir / raw_model_path).resolve())
        else:
            obj.model_path = raw_model_path
        obj.classes = meta["classes"]
        obj.classes_flattened = meta["classes_flattened"]
        obj.min_domain_count = meta.get("min_domain_count", 0)
        obj.gamma = meta.get("gamma", 1.0)
        obj.tau = meta.get("tau", 0.05)
        obj.margin = meta.get("margin", 0.10)
        obj.anchor_to_global = meta.get("anchor_to_global", True)
        obj.encoder_file = None  # auto-detect at inference time
        obj.probabilities = []
        obj.selected_classes = []
        obj._encoder = None
        obj._domain_bitmap = None
        obj._active_domain_mask = None
        obj._domain_train_ids = None

        obj.index = faiss.read_index(str(index_dir / "index.faiss"))
        if isinstance(obj.index, faiss.IndexIVF):
            obj.index.nprobe = obj.nprobe

        all_label_ids = np.load(index_dir / "label_ids.npy")
        obj.level_label_ids = [all_label_ids[i] for i in range(all_label_ids.shape[0])]

        all_class_names = np.load(index_dir / "class_names.npy", allow_pickle=True)
        obj.level_class_names = list(all_class_names)

        ctids_path = index_dir / "class_to_train_ids.pkl"
        if ctids_path.exists():
            with open(ctids_path, "rb") as f:
                obj.class_to_train_ids = pickle.load(f)
        else:
            obj.class_to_train_ids = []

        return obj

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _batch_search_scoped(
        self,
        index: "faiss.IndexFlatIP",
        batch_queries: np.ndarray,
        train_ids: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch FAISS search restricted to a subset of training vectors."""
        n_total = index.ntotal
        bitmap = np.zeros((n_total + 7) // 8, dtype=np.uint8)
        ids64 = train_ids.astype(np.int64)
        np.bitwise_or.at(bitmap, ids64 >> 3, (1 << (ids64 & 7)).astype(np.uint8))
        sel = faiss.IDSelectorBitmap(n_total, faiss.swig_ptr(bitmap))

        if isinstance(index, faiss.IndexIVF):
            params = faiss.SearchParametersIVF()
        else:
            params = faiss.SearchParameters()
        params.sel = sel
        scores, ids = index.search(batch_queries, k, params=params)  # type: ignore[call-arg]
        dist = np.arccos(np.clip(scores, -1.0, 1.0)) / np.pi
        dist[ids < 0] = np.inf
        return ids, dist

    def _gather_level_candidates(
        self,
        index: "faiss.IndexFlatIP",
        batch_queries: np.ndarray,
        sel_b: np.ndarray,
        cand_ids_l1: np.ndarray,
        cand_dist_l1: np.ndarray,
        lvl: int,
        n_batch: int,
    ) -> tuple[list, list, list]:
        """For lvl > 1: group docs by parent-class selection, do one batch search per group."""
        from collections import defaultdict

        lvl_ids: list = [None] * n_batch
        lvl_dist: list = [None] * n_batch
        lvl_active: list = [None] * n_batch

        prev_class_names = self.level_class_names[lvl - 2]

        groups: dict = defaultdict(list)
        for i in range(n_batch):
            sel = sel_b[i]
            key = tuple(sorted(sel[sel != ""]))
            groups[key].append(i)

        for parent_key, doc_indices in groups.items():
            active_class_mask = np.isin(
                self._get_classes(self.classes, lvl - 1), list(parent_key)
            )

            if active_class_mask.any() and self.class_to_train_ids:
                parent_ids_selected = np.where(
                    np.isin(prev_class_names, list(parent_key))
                )[0]
                parent_train_ids = (
                    np.concatenate([
                        self.class_to_train_ids[lvl - 2][pid]
                        for pid in parent_ids_selected
                    ]).astype(np.int64)
                    if len(parent_ids_selected) else np.array([], dtype=np.int64)
                )

                if len(parent_train_ids) > 0:
                    n_active = int(active_class_mask.sum())
                    K_level = min(self.k * n_active * 10, len(parent_train_ids))
                    batch_q = batch_queries[doc_indices]
                    ids_b, dist_b = self._batch_search_scoped(index, batch_q, parent_train_ids, K_level)
                    for j, i in enumerate(doc_indices):
                        valid = ids_b[j] >= 0
                        lvl_ids[i] = ids_b[j][valid]
                        lvl_dist[i] = dist_b[j][valid]
                        lvl_active[i] = active_class_mask
                    continue

            for i in doc_indices:
                parent_ids_sel = np.where(
                    np.isin(prev_class_names, list(parent_key))
                )[0].astype(np.uint32)
                prev_label_ids = self.level_label_ids[lvl - 2]
                mask = np.isin(prev_label_ids[cand_ids_l1[i]], parent_ids_sel)
                lvl_ids[i] = cand_ids_l1[i][mask]
                lvl_dist[i] = cand_dist_l1[i][mask]
                lvl_active[i] = active_class_mask

        return lvl_ids, lvl_dist, lvl_active

    def set_active_domains(self, domains: list[str] | None) -> None:
        """Restrict L1 search to a subset of domains.

        Pre-computes a FAISS IDSelectorBitmap over the training vectors that
        belong to the specified domains, so every subsequent predict call
        searches only within that subset at L1.  L2 (intent-level) search
        is automatically scoped to the domain(s) selected at L1.

        Pass None to clear the filter and search the full index.

        Typical usage in an OVOS plugin::

            domains = list({i.split(":")[0] for i in self.intents if ":" in i})
            self.model.set_active_domains(domains)
        """
        assert self.index is not None, "Call from_disk() before set_active_domains()."

        if not domains or not self.class_to_train_ids:
            self._domain_bitmap = None
            self._active_domain_mask = None
            self._domain_train_ids = None
            return

        domain_names = self.level_class_names[0]
        active_mask = np.isin(domain_names, list(domains))

        if not active_mask.any():
            self._domain_bitmap = None
            self._active_domain_mask = None
            self._domain_train_ids = None
            return

        train_ids = np.concatenate([
            self.class_to_train_ids[0][i]
            for i in np.where(active_mask)[0]
        ]).astype(np.int64)

        n_total = self.index.ntotal
        bitmap = np.zeros((n_total + 7) // 8, dtype=np.uint8)
        np.bitwise_or.at(bitmap, train_ids >> 3, (1 << (train_ids & 7)).astype(np.uint8))

        self._domain_bitmap = bitmap
        self._active_domain_mask = active_mask
        self._domain_train_ids = train_ids

    def predict_proba(self, documents, level: int | None = None, batch_size: int = 512):
        assert self.index is not None, "Call build() or from_disk() first."
        index: faiss.IndexFlatIP = self.index  # type: ignore[assignment]

        if isinstance(documents, np.ndarray):
            queries = np.ascontiguousarray(documents, dtype=np.float32)
            faiss.normalize_L2(queries)
        else:
            print("Encoding documents…")
            queries = self._encode_queries(documents, show_progress_bar=True)

        if queries.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch: queries have dim={queries.shape[1]} "
                f"but the index was built with dim={index.d}. "
                f"Make sure the encoder used to produce the embeddings matches "
                f"the one used at build time (model_path='{self.model_path}')."
            )

        max_level = level or self.get_depth()
        num_docs = len(queries)
        num_train = index.ntotal

        has_domain_filter = self._domain_bitmap is not None
        n_active_l1 = (
            int(self._active_domain_mask.sum()) if has_domain_filter   # type: ignore[union-attr]
            else len(self.level_class_names[0])
        )
        n_search_pool = len(self._domain_train_ids) if has_domain_filter else num_train  # type: ignore[arg-type]
        K_max = min(self.k * n_active_l1 * 10, n_search_pool)
        print(f"Predicting {num_docs:,} documents (K_max={K_max:,}, batch={batch_size})…")

        lvl_probabilities: list[list[np.ndarray]] = [[] for _ in range(max_level)]
        lvl_selected: list[list[np.ndarray]] = [[] for _ in range(max_level)]
        all_sel = np.full((num_docs, self.n), "", dtype=self.classes_flattened.dtype)
        all_prior = np.ones((num_docs, self.n))

        for batch_start in tqdm(range(0, num_docs, batch_size), desc="predicting"):
            batch_end = min(batch_start + batch_size, num_docs)
            batch_q = queries[batch_start:batch_end]
            n_batch = batch_end - batch_start

            # Unfiltered k=1 probe — gets the true nearest-neighbor distance in the
            # full index before any domain restriction, used as the margin/decay anchor.
            if self.anchor_to_global and has_domain_filter:
                anc_scores, _ = index.search(batch_q, 1)  # type: ignore[call-arg]
                d_anchor_b = np.arccos(np.clip(anc_scores[:, 0], -1.0, 1.0)) / np.pi

            # Main L1 search — restricted to active domains when a filter is set.
            if has_domain_filter:
                sel = faiss.IDSelectorBitmap(
                    index.ntotal, faiss.swig_ptr(self._domain_bitmap)
                )
                params = (
                    faiss.SearchParametersIVF()
                    if isinstance(index, faiss.IndexIVF)
                    else faiss.SearchParameters()
                )
                params.sel = sel
                scores_b, cids_b = index.search(batch_q, K_max, params=params)  # type: ignore[call-arg]
            else:
                scores_b, cids_b = index.search(batch_q, K_max)  # type: ignore[call-arg]

            cdist_b = np.arccos(np.clip(scores_b, -1.0, 1.0)) / np.pi
            if not (self.anchor_to_global and has_domain_filter):
                d_anchor_b = cdist_b[:, 0].copy()
            del scores_b

            sel_b = np.full((n_batch, self.n), "", dtype=self.classes_flattened.dtype)
            prior_b = np.ones((n_batch, self.n))
            prev_classes_b: np.ndarray = np.array([])
            batch_probs: list[np.ndarray] = []
            batch_sel: list[np.ndarray] = []

            for lvl in range(1, max_level + 1):
                label_ids = self.level_label_ids[lvl - 1]
                class_names = self.level_class_names[lvl - 1]
                probs_b = np.zeros((n_batch, len(class_names)))

                if lvl == 1:
                    l_ids: list = list(cids_b)
                    l_dist: list = list(cdist_b)
                    l1_active = (
                        self._active_domain_mask
                        if has_domain_filter
                        else np.ones(len(class_names), dtype=bool)
                    )
                    l_active: list = [l1_active] * n_batch
                else:
                    l_ids, l_dist, l_active = self._gather_level_candidates(
                        index, batch_q, sel_b, cids_b, cdist_b, lvl, n_batch,
                    )

                for i in range(n_batch):
                    ids = l_ids[i]
                    dist = l_dist[i]
                    active_class_mask = l_active[i]

                    if len(ids) == 0:
                        sel_b[i, :] = class_names[np.argsort(1.0 - probs_b[i])[: self.n]]
                        continue

                    active_classes = class_names[active_class_mask]
                    cand_labels = class_names[label_ids[ids]]

                    n_active = int(active_class_mask.sum())
                    K_local = min(self.k * n_active * 10, len(ids))
                    top_local = np.argpartition(dist, min(K_local - 1, len(dist) - 1))[:K_local]
                    cand_labels_trunc = cand_labels[top_local]
                    dist_trunc = dist[top_local]

                    cond_probs = np.zeros(len(class_names))
                    cond_probs[active_class_mask] = self._get_probabilities_with_adaptive_neighborhood(
                        distances=dist_trunc,
                        classes=active_classes,
                        labels=cand_labels_trunc,
                        d_anchor=float(d_anchor_b[i]) if self.anchor_to_global else None,
                    )

                    if lvl > 1:
                        parent_names = np.array(
                            [self._get_previous_level(c, self.classes, lvl) for c in class_names]
                        )
                        prior_probs = np.zeros(len(class_names))
                        prev_probs = batch_probs[-1][i]
                        for ci, parent in enumerate(parent_names):
                            if parent is not None and parent in prev_classes_b:
                                pidx = np.where(prev_classes_b == parent)[0]
                                if len(pidx):
                                    prior_probs[ci] = prev_probs[pidx[0]]
                        prior_probs[active_class_mask] = (
                            prior_probs[active_class_mask] / prior_probs[active_class_mask].sum()
                            if prior_probs[active_class_mask].sum() > 0
                            else 1.0 / active_class_mask.sum()
                        )
                    else:
                        prior_probs = np.ones(len(class_names)) / len(class_names)

                    post = prior_probs * cond_probs
                    total = post.sum()
                    probs_b[i] = post / total if total > 0 else post

                    top_n = np.argsort(1.0 - probs_b[i])[: self.n]
                    sel_b[i, :] = class_names[top_n]
                    prior_b[i, :] = probs_b[i, top_n]

                batch_probs.append(probs_b)
                batch_sel.append(sel_b.copy())
                prev_classes_b = class_names.copy()

            for lvl_idx in range(max_level):
                lvl_probabilities[lvl_idx].append(batch_probs[lvl_idx])
                lvl_selected[lvl_idx].append(batch_sel[lvl_idx])

            all_sel[batch_start:batch_end] = sel_b
            all_prior[batch_start:batch_end] = prior_b

        self.probabilities = [np.concatenate(p, axis=0) for p in lvl_probabilities]
        self.selected_classes = [np.concatenate(s, axis=0) for s in lvl_selected]

        return [
            dict(zip(map(str, c_vals), map(float, p_vals)))
            for c_vals, p_vals in zip(all_sel, all_prior)
        ]

    def predict(self, documents, level: int | None = None) -> list[str]:
        _ = self.predict_proba(documents, level=level)
        return self.selected_classes[-1][:, 0].tolist()

