"""
Encoder abstraction used by the classifier.

Three implementations are provided:
  StaticModelEncoder    — wraps model2vec StaticModel (fast, low-memory, dim=64)
  EmbeddingGemmaEncoder — EmbeddingGemma ONNX (Q4_0) with document/query prefixes, dim=768
  GraniteEncoder        — IBM Granite ONNX, CLS pooling, no prefixes, dim=768

load_encoder(model_path) auto-detects which one to instantiate by directory layout:
  • <path>/onnx/model_q4.onnx present           → EmbeddingGemmaEncoder
  • <path>/onnx/model_quint8_avx2.onnx present  → GraniteEncoder (quantised, AVX2)
  • <path>/onnx/model_uint8.onnx present        → GraniteEncoder (quantised)
  • <path>/onnx/model.onnx present              → GraniteEncoder (full F32)
  • otherwise                                   → StaticModelEncoder

Pass onnx_filename to force a specific ONNX file, e.g. "model.onnx" at build time
for maximum-precision embeddings.
"""

from __future__ import annotations

import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm


class StaticModelEncoder:
    """Wraps model2vec StaticModel. No prefix distinction between queries and documents."""

    def __init__(self, model_path: str) -> None:
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(model_path)

    @property
    def dim(self) -> int:
        return self._model.dim  # type: ignore[return-value]

    def _encode(self, texts: list[str], show_progress_bar: bool) -> np.ndarray:
        embs = self._model.encode(texts, show_progress_bar=show_progress_bar)
        embs = embs.astype(np.float32)
        faiss.normalize_L2(embs)
        return embs

    def encode_documents(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress_bar)

    def encode_queries(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress_bar)


class EmbeddingGemmaEncoder:
    """
    Wraps the EmbeddingGemma ONNX model (Q4_0 quantised).

    Documents and queries use different task prefixes as required by the model:
      encode_documents → "title: none | text: <text>"
      encode_queries   → "task: search result | query: <text>"
    """

    DOCUMENT_PREFIX = "title: none | text: "
    QUERY_PREFIX = "task: search result | query: "
    ONNX_FILENAME = "model_q4.onnx"

    def __init__(self, model_path: str, batch_size: int = 512, num_threads: int = 4) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        p = Path(model_path)
        onnx_path = p / "onnx" / self.ONNX_FILENAME

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        self._batch_size = batch_size
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self._run_batch(["hello"])
            self._dim = probe.shape[1]
        return self._dim

    def _run_batch(self, texts: list[str]) -> np.ndarray:
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        _, embs = self._session.run(None, inputs.data)
        return embs.astype(np.float32)

    def _encode(self, texts: list[str], show_progress_bar: bool) -> np.ndarray:
        batches = range(0, len(texts), self._batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="encoding")  # type: ignore[assignment]
        chunks = [self._run_batch(texts[i : i + self._batch_size]) for i in batches]
        result = np.concatenate(chunks, axis=0)
        faiss.normalize_L2(result)
        return result

    def encode_documents(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode([self.DOCUMENT_PREFIX + t for t in texts], show_progress_bar)

    def encode_queries(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode([self.QUERY_PREFIX + t for t in texts], show_progress_bar)


class GraniteEncoder:
    """
    Wraps IBM Granite embedding ONNX models.

    Uses the fast `tokenizers` library for batched tokenisation and CLS-token
    pooling. No task prefix distinction between documents and queries.

    Known variants detected by load_encoder:
      model_quint8_avx2.onnx  — granite-97m-r2
      model_uint8.onnx        — granite-107m-onnx
    """

    ONNX_FILENAMES = ("model_quint8_avx2.onnx", "model_uint8.onnx", "model.onnx")

    def __init__(
        self,
        model_path: str,
        onnx_filename: str = "model_quint8_avx2.onnx",
        batch_size: int = 512,
        num_threads: int = 4,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        p = Path(model_path)
        onnx_path = p / "onnx" / onnx_filename

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(p / "tokenizer.json"))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=256)
        self._batch_size = batch_size
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self._run_batch(["hello"])
            self._dim = probe.shape[1]
        return self._dim

    def _run_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        outputs = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        last_hidden_state = outputs[0]          # (batch, seq_len, hidden_size)
        cls_embs = last_hidden_state[:, 0, :]   # CLS token
        return cls_embs.astype(np.float32)

    def _encode(self, texts: list[str], show_progress_bar: bool) -> np.ndarray:
        batches = range(0, len(texts), self._batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="encoding")  # type: ignore[assignment]
        chunks = [self._run_batch(texts[i : i + self._batch_size]) for i in batches]
        result = np.concatenate(chunks, axis=0)
        faiss.normalize_L2(result)
        return result

    def encode_documents(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress_bar)

    def encode_queries(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress_bar)


AnyEncoder = StaticModelEncoder | EmbeddingGemmaEncoder | GraniteEncoder


def load_encoder(model_path: str, onnx_filename: str | None = None) -> AnyEncoder:
    """Return the right encoder for model_path.

    When onnx_filename is given the auto-detection is skipped and that specific
    ONNX file is loaded as a GraniteEncoder.  Use this to force full-precision
    embeddings at build time (e.g. onnx_filename="model.onnx").

    Auto-detection order (when onnx_filename is None):
      model_q4.onnx          → EmbeddingGemmaEncoder
      model_quint8_avx2.onnx → GraniteEncoder
      model_uint8.onnx       → GraniteEncoder
      model.onnx             → GraniteEncoder
      (none of the above)    → StaticModelEncoder
    """
    p = Path(model_path)
    if onnx_filename is not None:
        onnx_path = p / "onnx" / onnx_filename
        if not onnx_path.exists():
            raise FileNotFoundError(f"Requested encoder file not found: {onnx_path}")
        return GraniteEncoder(model_path, onnx_filename=onnx_filename)
    if (p / "onnx" / EmbeddingGemmaEncoder.ONNX_FILENAME).exists():
        return EmbeddingGemmaEncoder(model_path)
    for fn in GraniteEncoder.ONNX_FILENAMES:
        if (p / "onnx" / fn).exists():
            return GraniteEncoder(model_path, onnx_filename=fn)
    return StaticModelEncoder(model_path)

