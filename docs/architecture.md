# Architecture

This document explains how the classifier and pipeline plugin work from the inside.

---

## Overview

```
User utterance
      │
      ▼
 GraniteEncoder             ← ONNX model, ~94 MB
      │  float32 embedding (384-dim, L2-normalised)
      ▼
HierarchicalPairKNNClassifier
  ├─ L1 search  ──── top-N domains (skill namespaces)
  └─ L2 search  ──── top-k intents within those domains
      │  Wu-Lin probability distribution
      ▼
 Intent filter              ← only keep loaded-skill intents
      │  optional renormalisation
      ▼
 IntentHandlerMatch         ← returned to OVOS
```

---

## Two-phase hierarchy

Labels in the index follow a `domain:intent` format (e.g. `weather:get_weather`, `ocp:play`). The classifier exploits this structure to run two independent k-NN searches rather than one flat search over all intents.

### L1 — Domain search

The full FAISS index is queried for the `k` nearest neighbours of the query embedding. The top-`n` unique domains (left-hand side of `:`) are selected by probability mass. Domain pre-filtering (`set_active_domains`) can restrict L1 to only the domains of currently loaded skills, which reduces noise when few skills are installed.

### L2 — Intent search

For each of the top-`n` domains selected in L1, the same FAISS index is searched again but restricted to training vectors belonging to that domain. The top-`k` neighbours within each domain are retrieved and fed into the Wu-Lin estimator to produce a per-intent probability.

Final probabilities are the product of domain probability (L1) × intent probability (L2 | domain).

---

## Wu-Lin probability estimation

Raw k-NN distances are converted to probabilities using a pairwise kernel method derived from Wu & Lin (2004).

### Basic form

Each neighbour at distance `d` contributes a decayed weight:

```
w = exp(-γ · d)
```

Weights are accumulated per class and normalised to sum to 1.

### Adaptive neighbourhood (default)

The standard form is sensitive to the scale of distances, which varies with domain density. The adaptive form anchors the margin and decay to a reference distance `d_anchor`:

1. **Exact-match override**: if the nearest neighbour distance ≤ `tau` (default 0.05), the nearest class wins 100%.
2. **Adaptive margin**: only neighbours within `[d_anchor, d_anchor + margin]` are considered. Neighbours closer than `d_anchor` are always included; neighbours beyond `d_anchor + margin` are excluded.
3. **Distance decay**: weights use a Gaussian kernel anchored at `d_anchor`, keeping the effective decay invariant to domain density.
4. **Dynamic k**: the effective neighbourhood size scales with the number of active classes and the distance spread, rather than being a fixed hyperparameter.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `tau` | `0.05` | Exact-match threshold. Lower = more selective winner-takes-all. |
| `margin` | `0.10` | Shell half-width. Wider = smoother distributions. |
| `gamma` | `1.0` | Decay rate. Higher = sharper falloff with distance. |
| `anchor_to_global` | `True` | Anchor margin to the global nearest neighbour (not domain-filtered). Prevents bias when domains have very different densities. |

---

## FAISS index

The index is built with `IndexIVFPQ`:

- **Quantiser:** `IndexFlatIP` (inner product, equivalent to cosine similarity for L2-normalised vectors)
- **IVF:** 1024 Voronoi cells (`nlist`)
- **PQ:** 16 sub-quantisers, 8 bits each (`pq_m`)
- **Search:** 32 cells probed per query (`nprobe`)

For small datasets (fewer than `nlist × 256` vectors) a flat `IndexFlatIP` is used instead, which is exact but slower for large collections.

Vectors are L2-normalised before insertion so inner product equals cosine similarity.

---

## Intent filtering and renormalisation

After the classifier returns a probability distribution over all labels in the index, the pipeline applies two filters:

1. **Registered-intent filter**: only labels whose skill is currently loaded (via Adapt or Padatious manifest) pass through. Unrecognised labels are discarded.
2. **Ignore list**: labels listed in `ignore_intents` are discarded.

Special labels bypass the registered-intent filter but are gated by the session pipeline configuration:

| Label | Required pipeline stage |
|---|---|
| `ocp:play` | `ocp` in session pipeline |
| `common_query:common_query` | `common_query` in session pipeline |
| `stop:stop` | `stop` in session pipeline |

After filtering, if `renormalize=true`, surviving probabilities are re-scaled to sum to 1.

---

## Dynamic intent synchronisation

The pipeline maintains an allowlist of registered intents that is updated in real time:

- **At startup**: queries the Adapt and Padatious services for all currently registered intent names.
- **On skill load**: listens for `intent.service.intent-registered` and similar bus messages.
- **On skill detach**: removes the skill's intents from the allowlist.

This ensures that intents from skills that load after the pipeline starts are immediately available, and intents from unloaded skills are immediately blocked.

---

## File structure of a built index

```
index_dir/
├── index.faiss              ← FAISS IVF+PQ index
├── label_ids.npy            ← label-to-class-ID mapping per hierarchy level
├── class_names.npy          ← class name array per hierarchy level
├── class_to_train_ids.pkl   ← training vector IDs per class (for scoped L2 search)
└── meta.pkl                 ← classifier hyperparameters + model_path
```

The encoder model is stored separately (either inside the same directory or referenced by `meta.pkl → model_path`).

---

## Memory and latency budget (Raspberry Pi 4)

| Component | Size | Load time | Inference latency |
|---|---|---|---|
| Granite ONNX encoder | ~94 MB | ~2 s | ~30 ms/utterance |
| FAISS IVF+PQ index | ~466 MB | ~1 s | <5 ms/query |
| **Total** | **~560 MB** | **~3 s** | **~35 ms** |

These are approximate figures for a Pi 4 with 4 GB RAM. Actual performance varies with the number of loaded skills and FAISS `nprobe`.
