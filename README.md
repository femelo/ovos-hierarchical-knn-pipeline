[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-hierarchical-knn-pipeline)

# OVOS Hierarchical KNN Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by a two-stage hierarchical k-NN classifier backed by IBM Granite embeddings and a FAISS index.

This plugin uses a pre-built [FAISS](https://github.com/facebookresearch/faiss) index to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It only considers intents from loaded skills and ignores any labels from unregistered intents. This pipeline is ideal for use cases where other deterministic engines fail to provide a high-confidence match.

---

## ✨ Features

* ✅ Two-stage hierarchical search: domain-level then intent-level, using Wu-Lin pairwise probability estimation
* ✅ Plug-and-play integration with OVOS pipelines
* ✅ Multilingual support (en, pt, es, fr, it, de, nl, ca, gl, da, eu)
* ✅ Powered by IBM Granite Embedding 97M Multilingual R2 (quantised ONNX, ~94 MB)
* ✅ FAISS IVF+PQ index for fast, memory-efficient inference on edge devices
* ✅ Syncs Adapt and Padatious intents dynamically at runtime
* ✅ Only considers intents from loaded skills, ignoring unregistered labels
* ✅ Domain pre-filtering: search is automatically scoped to the domains of loaded skills

> 💡 The quantised encoder (`model_quint8_avx2.onnx`) requires an AVX2-capable CPU. Total footprint is ~560 MB (index + encoder).

---

## 📦 Installation

You can install the plugin via `pip`:

```bash
pip install ovos-hierarchical-knn-pipeline
```

---

## ⚙️ Configuration

Download the pre-built index and model from HuggingFace:

```bash
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2')
print(path)
"
```

Then add the following to your `mycroft.conf`:

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
      "index_dir": "/path/to/downloaded/index",
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": []
    }
  }
}
```

* `index_dir`: Path to the directory downloaded from HuggingFace (required).
* `conf_xxx`: Minimum confidence threshold for intent matching at each pipeline stage.
* `ignore_intents`: List of intents to suppress during matching.
* `renormalize`: Re-scale surviving probabilities to sum to 1 after filtering (default `false`).

> ⚠️ The FAISS index is pre-built on a fixed dataset and **cannot learn new skills** dynamically. Skills not covered by the index are still reachable through the Adapt and Padatious stages of the pipeline.

---

## 🧠 Usage

The `HierarchicalKNNIntentPipeline` class integrates with the OVOS intent system. It:

1. Receives an utterance (text).
2. Encodes it with the IBM Granite ONNX encoder.
3. Runs a two-stage hierarchical k-NN search: first across skill domains, then across intents within the top-scoring domains.
4. Filters out intents that are not part of the loaded skills.
5. Returns a match for the highest-confidence intent from the list of valid intents.

---

## 🧪 Tips

* Tune `conf_high`, `conf_medium`, and `conf_low` to control the confidence threshold at each pipeline stage.
* Use the `ignore_intents` list to filter out specific problematic intents from predictions.
* Syncing of Adapt and Padatious intents is done automatically at runtime via the OVOS message bus.
* Enable `renormalize: true` if you want probabilities to always sum to 1 after unregistered intents are filtered out.

> 💡 Pre-built index available on HuggingFace: [fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2](https://huggingface.co/fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2)

---

## 🛡 License

This project is licensed under the [Apache 2.0 License](LICENSE).
