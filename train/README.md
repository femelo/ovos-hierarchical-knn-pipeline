# Training

Scripts for building a new index or retraining on a custom dataset.

## Pipeline

```
preprocess_data.py  →  balance_dataset.py  →  encode_dataset.py  →  build_index.py
```

`encode_dataset.py` is optional but strongly recommended: pre-computing embeddings once and storing them as a Parquet file avoids re-encoding during every build run.

## Scripts

| Script | Description |
|---|---|
| `preprocess_data.py` | Cleans a raw intent CSV: normalises label format to `domain:intent.intent`, fills `{placeholder}` slots with locale-aware Faker values, and extracts an `entity` column listing the slot types found in each sentence. |
| `balance_dataset.py` | Balances the class distribution: caps very large classes via FAISS k-means, downsamples mid-tier classes, and augments micro-tier classes first with Faker slot-filling and then with Gemini LLM paraphrases (requires `GEMINI_API_KEY`). Outputs a Parquet file with pre-computed embeddings. |
| `encode_dataset.py` | Encodes a CSV with the embedding model and saves as Parquet with `dim_0 … dim_N` columns. Feed the output directly to `build_index.py` to skip the encoding step at build time. |
| `create_diverse_subset.py` | Selects the most semantically diverse sentences per label using a greedy farthest-point (maximin) algorithm in embedding space. Useful for producing a compact, high-quality training set from a large corpus. |
| `detect_multimodality.py` | Analyses intra-class spread and runs k-means sub-clustering to detect labels whose training examples split into distinct semantic islands. Use the report to identify labels that may need splitting or extra data. |
| `build_index.py` | Builds the FAISS IVF+PQ index from a CSV or pre-encoded Parquet file and writes it to disk. This is the artefact loaded by the OVOS plugin at inference time. |

## Step-by-step

### 1 — Preprocess

Normalise labels and fill placeholder slots:

```bash
python train/preprocess_data.py \
    dataset/raw_intents.csv \
    dataset/intents_clean.csv
```

### 2 — Balance (optional)

Cap large classes and augment small ones. Embeddings are computed and stored in the output Parquet:

```bash
python train/balance_dataset.py \
    --input   dataset/intents_clean.csv \
    --output  dataset/intents_balanced.parquet \
    --model   /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

### 3 — Encode (optional, recommended)

If you skipped `balance_dataset.py`, pre-encode the cleaned CSV so `build_index.py` does not re-encode:

```bash
python train/encode_dataset.py \
    --input   dataset/intents_clean.csv \
    --output  dataset/intents_encoded.parquet \
    --model   /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

### 4 — Build the index

Pass a Parquet with pre-computed embeddings (recommended) or a plain CSV:

```bash
python train/build_index.py \
    --dataset   dataset/intents_balanced.parquet \
    --index-dir models/index \
    --model     /path/to/granite-97m-r2
```

When building from CSV the encoder is loaded automatically; use `--encoder-file` to select the ONNX variant (default: `model.onnx` for full F32 precision):

```bash
python train/build_index.py \
    --dataset      dataset/intents_clean.csv \
    --index-dir    models/index \
    --model        /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

### Key `build_index.py` options

| Option | Default | Description |
|---|---|---|
| `--model` | `models/granite-97m-onnx` | Path to the embedding model directory. |
| `--encoder-file` | `model.onnx` | ONNX file used for encoding (CSV path only). |
| `--langs` | 11 EU langs | Languages to include from the dataset. |
| `--min-count` | `10` | Minimum examples per label; rarer labels are dropped. |
| `--k` | `5` | Neighbours retrieved per Wu-Lin estimation. |
| `--n` | `4` | Top-n domains passed from L1 to L2 search. |
| `--nlist` | `1024` | Number of IVF centroids. |
| `--pq-m` | `16` | PQ sub-quantisers (bytes per vector). |
| `--nprobe` | `32` | IVF cells probed at search time. |
| `--margin` | `0.1` | Shell half-width for the adaptive neighbourhood. |
| `--gamma` | `1.0` | Decay rate for distance weighting (`0` = uniform). |

## Training dependencies

```bash
pip install -r train/requirements.txt
```
