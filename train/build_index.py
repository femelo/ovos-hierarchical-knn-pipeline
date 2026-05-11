"""
Offline preprocessing: build a FAISS IVF+PQ index from the training dataset.

Usage (CSV — embeddings computed on the fly)
--------------------------------------------
    python train/build_index.py \
        --dataset  dataset/intents_dataset.csv \
        --index-dir models/index \
        --model     models/granite-97m-onnx \
        --langs     en pt es fr it de ca \
        --min-count 3 \
        --nlist     1024 \
        --pq-m      16 \
        --nprobe    32 \
        --k         5 \
        --n         4

Usage (Parquet — pre-computed embeddings in dim_0 … dim_N columns)
------------------------------------------------------------------
    python train/build_index.py \
        --dataset   dataset/intents_dataset.parquet \
        --index-dir models/index \
        --model     models/granite-97m-onnx

    Required columns: domain, intent, sentence, lang, dim_0 … dim_N
    When dim_* columns are present the vectors are used directly (no
    encoding step). --model must still be provided: it is saved in the
    index metadata so that inference (from_disk) loads the correct model.

Memory targets (dim=64)
-----------------------
  pq_m=16 → 16 bytes/vector → ~32 MB for 2M vectors
  pq_m=8  →  8 bytes/vector → ~16 MB  (lower recall)
"""

import argparse
import re

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="dataset/merged_intents_dataset_full.csv")
    p.add_argument("--index-dir", default="models/edge_index")
    p.add_argument("--model",     default="models/granite-97m-onnx",
                   help="Embedding model path. Saved in index metadata and used at inference. "
                        "When building from a Parquet file with pre-computed embeddings, "
                        "pass the model that was used to produce those embeddings.")
    p.add_argument("--langs",     nargs="+", default=["en", "pt", "es", "fr", "it", "de", "nl", "ca", "gl", "da", "eu"])
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--nlist",     type=int, default=1024)
    p.add_argument("--pq-m",      type=int, default=16)
    p.add_argument("--nprobe",    type=int, default=32)
    p.add_argument("--k",         type=int,   default=5)
    p.add_argument("--n",         type=int,   default=4)
    p.add_argument("--gamma",     type=float, default=1.0,
                   help="Decay rate for distance weighting in Wu-Lin: w=exp(-gamma*d). "
                        "Higher values down-weight distant neighbors more aggressively.")
    p.add_argument("--tau",       type=float, default=0.5,
                   help="Threshold for the similarity score in Wu-Lin: s = 1 - exp(-tau * d). "
                        "Higher values make the similarity score more sensitive to distance.")
    p.add_argument("--margin",    type=float, default=0.1,
                   help="Shell half-width around the nearest neighbor for the adaptive neighborhood.")
    p.add_argument("--anchor-to-global", action=argparse.BooleanOptionalAction, default=True,
                   help="Anchor the margin shell and decay to the global (unfiltered) nearest "
                        "neighbor distance rather than the nearest candidate in the pool. "
                        "Disable with --no-anchor-to-global to test the effect in isolation.")
    p.add_argument("--encoder-file", default="model.onnx",
                   help="ONNX filename to use for encoding when building from CSV "
                        "(e.g. 'model.onnx' for full F32 precision, "
                        "'model_quint8_avx2.onnx' for quantised). "
                        "Ignored when building from Parquet with pre-computed embeddings.")
    p.add_argument("--sanity-only", action="store_true",
                   help="Skip build and only run the sanity check on an existing index.")
    return p.parse_args()


def load_dataset(path: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load CSV or Parquet. Returns (df, embeddings_or_None).

    For Parquet files that contain dim_0…dim_N columns the embedding matrix
    is extracted and returned separately; those columns are dropped from df.
    CSV files always return None for embeddings (encoded during build).
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        emb_cols = sorted(
            [c for c in df.columns if re.match(r"^dim_\d+$", c)],
            key=lambda c: int(c.split("_")[1]),
        )
        if emb_cols:
            print(f"  Found {len(emb_cols)} pre-computed embedding columns.")
            embeddings = df[emb_cols].to_numpy(dtype=np.float32)
            df = df.drop(columns=emb_cols)
        else:
            embeddings = None
    else:
        df = pd.read_csv(p)
        embeddings = None

    return df, embeddings


def main():
    args = parse_args()

    print("Loading dataset…")
    df, embeddings = load_dataset(args.dataset)

    has_label_col = "label" in df.columns
    required_cols = ["label", "sentence"] if has_label_col else ["domain", "intent", "sentence"]

    before = len(df)
    df = df.dropna(subset=required_cols)
    if embeddings is not None:
        embeddings = embeddings[df.index]
    df = df.reset_index(drop=True)

    if embeddings is None:
        # Strip placeholder tokens (e.g. {location}) that skew SIF embeddings.
        df["sentence"] = df["sentence"].str.replace(r"\{[^}]+\}", "", regex=True).str.strip()
        mask = df["sentence"].str.len() > 0
        df = df[mask].reset_index(drop=True)

    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with null values.")

    if not has_label_col:
        df["label"] = df["domain"] + ":" + df["intent"]
    else:
        df["domain"] = df["label"].str.split(":").str[0]
        df["intent"] = df["label"].str.split(":").str[1]
    lang_mask = df["lang"].isin(args.langs)
    df = df[lang_mask].reset_index(drop=True)
    if embeddings is not None:
        embeddings = embeddings[lang_mask.values]

    counts = df["label"].value_counts()
    valid_labels = counts[counts >= args.min_count].index
    label_mask = df["label"].isin(valid_labels)
    df = df[label_mask].reset_index(drop=True)
    if embeddings is not None:
        embeddings = embeddings[label_mask.values]

    print(f"  {len(df):,} sentences, {df['label'].nunique()} intents, {df['domain'].nunique()} domains")

    if not args.sanity_only:
        labels = df["label"].unique().tolist()
        clf = HierarchicalPairKNNClassifier(
            classes=labels,
            k=args.k,
            n=args.n,
            nlist=args.nlist,
            pq_m=args.pq_m,
            nprobe=args.nprobe,
            model_path=args.model,
            gamma=args.gamma,
            tau=args.tau,
            margin=args.margin,
            anchor_to_global=args.anchor_to_global,
            encoder_file=args.encoder_file if embeddings is None else None,
        )
        clf.build(
            df["sentence"].tolist(),
            df["label"].tolist(),
            args.index_dir,
            embeddings=embeddings,
        )

    # Quick sanity check
    print("\nSanity check…")
    samples = df.sample(5, random_state=0)
    clf2 = HierarchicalPairKNNClassifier.from_disk(args.index_dir)
    preds = clf2.predict(samples["sentence"].tolist())
    for sent, pred, true in zip(samples["sentence"], preds, samples["label"]):
        match = "✓" if pred == true else "✗"
        print(f"  {match}  [{true}]  →  {pred!r}  |  {sent!r}")


if __name__ == "__main__":
    main()

