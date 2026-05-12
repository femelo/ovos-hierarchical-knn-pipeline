# Troubleshooting

---

## No AVX2 support {#no-avx2}

**Symptom:** `onnxruntime` raises an `Illegal instruction` error or fails to load `model_quint8_avx2.onnx`.

**Cause:** The default quantised encoder requires AVX2 CPU instructions, which are available on most x86-64 CPUs since ~2013 but absent on some older hardware and ARM devices.

**Fix:** Force the non-AVX2 quantised variant:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": "/path/to/index"
    }
  }
}
```

Then, when loading the classifier manually, specify the fallback encoder:

```python
clf = HierarchicalPairKNNClassifier.from_disk("/path/to/index")
clf._encoder = load_encoder("/path/to/model", onnx_filename="model_uint8.onnx")
```

Or rebuild the index with `model_uint8.onnx` as the encoder file and update `meta.pkl` accordingly.

For ARM (Raspberry Pi), use the float32 variant (`model.onnx`) — it is slower but has no special CPU requirements.

---

## Index not found / download fails

**Symptom:** `FileNotFoundError` or network error on startup.

**Causes and fixes:**

1. **No internet access:** Download the snapshot manually and set `index_dir` (see [Getting Started — offline setup](getting-started.md#offline--air-gapped-setup)).

2. **HuggingFace rate limit:** The download is large (~560 MB). Retry after a few minutes or use `huggingface-cli login` to authenticate.

3. **Corrupted cache:** Delete the cache and re-download:

   ```bash
   rm -rf ~/.cache/huggingface/hub/models--fdemelo--ovos-hierarchical-knn-granite-97m-multilingual-r2
   ```

---

## No intents matched / all results below threshold

**Symptom:** Every utterance returns `None` from all three confidence tiers.

**Possible causes:**

1. **Skills not loaded yet:** The pipeline filters to registered intents only. If no skills have loaded when the utterance arrives, there are no valid intents and nothing matches. Check that skills are fully initialised before sending utterances.

2. **Skills not in training data:** The pre-built index was trained on a fixed skill set. Skills whose intents are absent from the index will not match. Use Adapt or Padatious for those skills, or [build a custom index](training.md).

3. **Thresholds too high:** Lower `conf_low` to `0.05` or `0.0` to always return the top prediction.

4. **Wrong language:** The index covers 11 European languages. Utterances in other languages will not match reliably.

---

## Intent matched to wrong skill

**Symptom:** The pipeline returns the right intent name but the wrong skill ID.

**Cause:** Skill IDs are derived from the domain portion of the label (left-hand side of `:`). If two skills share a domain prefix, the wrong skill may be targeted.

**Fix:** Add the conflicting intent label to `ignore_intents` and let Adapt or Padatious handle it:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "ignore_intents": ["conflicting_domain:conflicting_intent"]
    }
  }
}
```

---

## High memory usage

**Symptom:** OVOS OOMs on devices with less than 1 GB of RAM.

**Cause:** The FAISS IVF+PQ index (~466 MB) and Granite ONNX encoder (~94 MB) are loaded into RAM.

**Options:**

1. Build a pruned index covering only your installed skills (smaller dataset = smaller index).
2. Increase swap space on the device.
3. Use a lighter encoder (`StaticModelEncoder` via `model2vec`) — build a new index with it and point `model_path` accordingly.

---

## Pipeline stage fires unexpectedly

**Symptom:** The KNN pipeline matches an utterance that you expected another engine to handle.

**Fix:** Move the KNN stage to a later position in the pipeline list, after the deterministic engines:

```json
"pipeline": [
  "adapt-high",
  "padatious-high",
  "adapt-medium",
  "padatious-medium",
  "adapt-low",
  "padatious-low",
  "ovos-hierarchical-knn-pipeline-low",
  "fallback-low"
]
```

---

## `renormalize` changes confidence scores unexpectedly

**Symptom:** After enabling `renormalize: true`, confidence scores are higher than expected.

**Explanation:** Renormalisation re-distributes the probability mass from filtered (unregistered) intents to the surviving intents. If many intents were filtered out, the surviving probability mass is small and gets scaled up significantly. This is correct behaviour — the score reflects relative certainty among valid intents.

If you need raw scores for comparison, set `renormalize: false`.

---

## Startup is slow

**Symptom:** OVOS takes 5+ seconds longer to start after adding the plugin.

**Cause:** The encoder ONNX model is loaded on the first inference call (lazy init), not at startup. The delay you see is likely the first utterance triggering model load.

**Fix:** The model is loaded once and stays in memory. Subsequent inferences are fast. If startup time itself is the issue, it is likely the HuggingFace download on first run — see the offline setup instructions in [Getting Started](getting-started.md).
