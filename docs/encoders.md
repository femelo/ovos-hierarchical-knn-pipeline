# Encoders

The encoder converts text strings into fixed-dimensional float32 embeddings that are fed into the FAISS index. The library ships three encoder implementations and auto-detects which one to use based on the files present in the model directory.

---

## Auto-detection logic

```python
from ovos_hierarchical_knn_pipeline.encoders import load_encoder

enc = load_encoder("/path/to/model")
```

The detection order is:

1. If `onnx/model_q4.onnx` exists → `EmbeddingGemmaEncoder`
2. Else if `model_quint8_avx2.onnx`, `model_uint8.onnx`, or `model.onnx` exists → `GraniteEncoder`
3. Else → `StaticModelEncoder`

You can override auto-detection by passing `onnx_filename`:

```python
enc = load_encoder("/path/to/model", onnx_filename="model_uint8.onnx")
```

---

## `GraniteEncoder`

Wraps IBM Granite Embedding 97M Multilingual R2 in ONNX format.

- **Pooling:** CLS token
- **Dimension:** 384
- **No prefix distinction** between documents and queries
- **ONNX variants** (checked in order):
  - `model_quint8_avx2.onnx` — uint8 quantised, requires AVX2 CPU (fastest, default)
  - `model_uint8.onnx` — uint8 quantised, no AVX2 requirement
  - `model.onnx` — full float32 (largest, highest precision)
- **Tokeniser:** `tokenizer.json` in model directory (using the `tokenizers` library)

This is the default encoder for the pre-built index.

---

## `EmbeddingGemmaEncoder`

Wraps an EmbeddingGemma ONNX model (Q4_0 quantised).

- **Pooling:** varies by model
- **Dimension:** model-dependent
- **Prefix distinction:**
  - Documents: `"title: none | text: <text>"`
  - Queries: `"task: search result | query: <text>"`
- **Tokeniser:** loaded via HuggingFace `transformers`

Use this encoder when your model directory contains `onnx/model_q4.onnx`.

---

## `StaticModelEncoder`

Wraps a `model2vec` static embedding model.

- **Pooling:** mean pooling (no attention needed)
- **Dimension:** 64 (typical)
- **No prefix distinction**
- **Fastest** of the three; lowest precision

Auto-detected when no ONNX files are found. Useful for experimentation with smaller models.

---

## Protocol

All encoders implement the `AnyEncoder` protocol:

```python
class AnyEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode_documents(
        self, texts: list[str], show_progress_bar: bool = False
    ) -> np.ndarray: ...

    def encode_queries(
        self, texts: list[str], show_progress_bar: bool = False
    ) -> np.ndarray: ...
```

For `GraniteEncoder` and `StaticModelEncoder`, `encode_documents` and `encode_queries` are equivalent. For `EmbeddingGemmaEncoder`, they apply different prefixes.

At inference time the pipeline always calls `encode_queries`. At build time `build_index.py` calls `encode_documents`.

---

## Using the encoder directly

```python
from ovos_hierarchical_knn_pipeline.encoders import load_encoder

enc = load_encoder("/path/to/granite-97m-r2")
print(f"Embedding dimension: {enc.dim}")

embeddings = enc.encode_queries(["play some jazz", "what time is it"])
print(embeddings.shape)   # (2, 384)
print(embeddings.dtype)   # float32
```

Returned embeddings are L2-normalised.

---

## Downloading the default encoder

The encoder is bundled inside the HuggingFace snapshot downloaded by `from_pretrained()`. If you need just the encoder without the FAISS index:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ibm-granite/granite-embedding-97m-multilingual',
    local_dir='/opt/granite-97m-r2',
)
"
```
