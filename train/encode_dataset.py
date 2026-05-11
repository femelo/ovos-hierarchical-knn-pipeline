"""
Pre-encode a dataset CSV with a specified embedding model and save as Parquet.

Embedding dimensions are stored as individual columns dim_0, dim_1, …, dim_N,
matching the format expected by build_index.py and balance_dataset.py.

Usage
-----
    python train/encode_dataset.py \
        --input  dataset/test_set.csv \
        --output dataset/test_set_encoded.parquet \
        --model  models/granite-97m-onnx
"""

import argparse
import numpy as np
import pandas as pd
from ovos_hierarchical_knn_pipeline.encoders import load_encoder


def parse_args():
    p = argparse.ArgumentParser(
        description="Encode a CSV dataset and save as Parquet with dim_* embedding columns."
    )
    p.add_argument("--input",  required=True, help="Path to input CSV file.")
    p.add_argument("--output", required=True, help="Path to output Parquet file.")
    p.add_argument(
        "--model",
        default="models/granite-97m-onnx",
        help="Model path. Auto-detected: EmbeddingGemma if onnx/model_q4.onnx exists, "
             "Granite if onnx/model_quint8_avx2.onnx or onnx/model_uint8.onnx exists, "
             "otherwise model2vec StaticModel.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from {args.input}…")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows loaded.")

    # Drop any existing dim_* columns so we don't double-encode.
    existing_dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if existing_dim_cols:
        print(f"  Dropping {len(existing_dim_cols)} existing dim_* columns.")
        df = df.drop(columns=existing_dim_cols)

    print(f"Encoding with {args.model}…")
    encoder = load_encoder(args.model)
    embeddings = encoder.encode_documents(df["sentence"].tolist(), show_progress_bar=True)

    print("Expanding embeddings into dim_* columns…")
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"dim_{i}" for i in range(embeddings.shape[1])],
    )
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    output_path = args.output
    if not output_path.endswith(".parquet"):
        output_path = output_path.rsplit(".", 1)[0] + ".parquet"

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} rows × {embeddings.shape[1]} dims → {output_path}")


if __name__ == "__main__":
    main()

