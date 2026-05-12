# Configuration Reference

All options live under `intents.ovos_hierarchical_knn_pipeline` in `mycroft.conf`.

---

## Full example

```json
{
  "intents": {
    "pipeline": [
      "ovos-hierarchical-knn-pipeline-high",
      "adapt-high",
      "padatious-high",
      "ovos-hierarchical-knn-pipeline-medium",
      "adapt-medium",
      "padatious-medium",
      "ovos-hierarchical-knn-pipeline-low",
      "adapt-low",
      "padatious-low",
      "fallback-low"
    ],
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": null,
      "hf_repo_id": "fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2",
      "hf_cache_dir": null,
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": [],
      "renormalize": true,
      "timeout": 1
    }
  }
}
```

---

## Key reference

### `index_dir`

| | |
|---|---|
| Type | `string \| null` |
| Default | `null` (auto-download from HuggingFace) |

Path to a local directory containing a pre-built index. When set, `hf_repo_id` and `hf_cache_dir` are ignored. The directory must contain `index.faiss`, `label_ids.npy`, `class_names.npy`, `class_to_train_ids.pkl`, `meta.pkl`, and the encoder model files.

Use this for air-gapped deployments or when you have built a custom index.

---

### `hf_repo_id`

| | |
|---|---|
| Type | `string` |
| Default | `fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2` |

HuggingFace repository to download when `index_dir` is not set. Ignored when `index_dir` is set.

---

### `hf_cache_dir`

| | |
|---|---|
| Type | `string \| null` |
| Default | HuggingFace default (`~/.cache/huggingface/hub`) |

Local directory for the downloaded HuggingFace snapshot. Ignored when `index_dir` is set.

---

### `conf_high`

| | |
|---|---|
| Type | `float` |
| Default | `0.7` |
| Range | `0.0 – 1.0` |

Minimum confidence score required for a match returned by `match_high` (the `ovos-hierarchical-knn-pipeline-high` pipeline stage). Utterances that score below this threshold are passed to the next stage.

Increasing this value makes the high stage more conservative; decreasing it makes it more aggressive.

---

### `conf_medium`

| | |
|---|---|
| Type | `float` |
| Default | `0.5` |
| Range | `0.0 – 1.0` |

Minimum confidence for `match_medium` (the `ovos-hierarchical-knn-pipeline-medium` stage).

---

### `conf_low`

| | |
|---|---|
| Type | `float` |
| Default | `0.15` |
| Range | `0.0 – 1.0` |

Minimum confidence for `match_low` (the `ovos-hierarchical-knn-pipeline-low` stage). Set to `0.0` to always return the top prediction regardless of confidence.

---

### `ignore_intents`

| | |
|---|---|
| Type | `list[string]` |
| Default | `[]` |

List of intent labels to suppress. Labels in this list are treated as if they were not registered, so the pipeline skips them and considers the next best prediction.

Example — suppress the OCP play intent so it is always handled by the OCP pipeline stage:

```json
"ignore_intents": ["ocp:play"]
```

---

### `renormalize`

| | |
|---|---|
| Type | `bool` |
| Default | `true` |

When `true`, probabilities are re-scaled to sum to 1 after unregistered and ignored intents are filtered out. This ensures the returned confidence score reflects the relative certainty among valid intents only.

When `false`, the raw Wu-Lin probability is returned. The score will be lower than usual if many intents were filtered out, because the probability mass that was assigned to filtered intents is simply discarded.

**Recommendation:** leave `true` unless you need to compare raw scores across different filter configurations.

---

### `timeout`

| | |
|---|---|
| Type | `float` |
| Default | `1` |

Seconds to wait for a response when querying the Adapt and Padatious services for registered intent names at startup. Increase this on slow hardware or when loading many skills.

---

## Pipeline stage names

The plugin registers three pipeline stages:

| Stage name | Fires when |
|---|---|
| `ovos-hierarchical-knn-pipeline-high` | Confidence ≥ `conf_high` |
| `ovos-hierarchical-knn-pipeline-medium` | Confidence ≥ `conf_medium` |
| `ovos-hierarchical-knn-pipeline-low` | Confidence ≥ `conf_low` |

Include any combination of these in your `pipeline` list. Omitting a stage prevents the plugin from firing at that confidence tier.
