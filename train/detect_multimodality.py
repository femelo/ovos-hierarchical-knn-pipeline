import argparse
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suppress KMeans memory leak warning on Windows/small clusters
warnings.filterwarnings("ignore")


def detect_multimodal_classes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 20,
    max_eval_samples: int = 10000
) -> pd.DataFrame:

    unique_labels = np.unique(labels)
    results = []

    print(f"\nAnalyzing internal topology of {len(unique_labels)} classes...")

    for label in tqdm(unique_labels, desc="Analyzing Classes"):
        # Isolate embeddings for this specific intent
        class_mask = (labels == label)
        X = embeddings[class_mask]
        n_samples = len(X)

        # Prevent System RAM OOM on massive classes (like ocp:play)
        if n_samples > max_eval_samples:
            idx = np.random.choice(n_samples, max_eval_samples, replace=False)
            X_eval = X[idx]
            eval_count = max_eval_samples
        else:
            X_eval = X
            eval_count = n_samples

        # 1. Calculate the Intra-Class Spread (Average pairwise angular distance)
        # Needs at least 2 samples to have a spread.
        if eval_count > 1:
            sim_matrix = np.dot(X_eval, X_eval.T)
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            dist_matrix = np.arccos(sim_matrix) / np.pi

            upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
            mean_spread = float(np.mean(dist_matrix[upper_triangle_indices]))
        else:
            mean_spread = 0.0

        # 2. Estimate the number of semantic "Islands" (Sub-clusters)
        best_k = None
        best_silhouette = None

        # Only run K-Means if we meet the minimum sample threshold for reliable clustering
        if n_samples >= min_samples:
            best_k = 1
            temp_best_silhouette = -1.0

            # Try splitting the class into 2, 3, 4, or 5 sub-clusters
            max_k = min(6, eval_count // 5)

            if max_k > 2:
                for k in range(2, max_k):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_eval)

                    # Silhouette score measures how distinct the sub-clusters are
                    score = silhouette_score(X_eval, cluster_labels, metric='cosine')

                    if score > temp_best_silhouette:
                        temp_best_silhouette = score
                        best_k = k

                # If the best silhouette score is very low (< 0.15), it's just a noisy cloud.
                if temp_best_silhouette >= 0.15:
                    best_silhouette = temp_best_silhouette
                else:
                    best_k = 1

        results.append({
            "Intent": label,
            "Total_Samples": n_samples,
            "Spread_Score": round(mean_spread, 4),
            "Estimated_Islands": best_k,
            "Island_Separation_Score": round(best_silhouette, 3) if best_silhouette is not None else None
        })

    # Convert to DataFrame and sort by Spread Score (highest = most multimodal)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Spread_Score", ascending=False).reset_index(drop=True)

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Detect highly multimodal classes from pre-calculated embeddings.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset containing embeddings (Parquet format).")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the column containing labels/intents.")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples required to run KMeans clustering.")
    parser.add_argument("--max_eval_samples", type=int, default=10000, help="Cap to prevent RAM OOM on massive intents.")
    parser.add_argument("--output", type=str, default="class_topology_report.csv", help="Output path for the report.")

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)

    if args.label_col not in df.columns:
        raise ValueError(f"Dataset must contain a '{args.label_col}' column.")

    emb_cols = [col for col in df.columns if col.startswith("dim_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Ensure they are named 'dim_0', 'dim_1', etc.")

    print(f"Found {len(emb_cols)} embedding dimensions.")

    labels = df[args.label_col].values

    print("Extracting and normalizing embeddings...")
    embeddings = df[emb_cols].values.astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings_norm = embeddings / norms

    print(f"Loaded {len(embeddings_norm)} samples across {len(np.unique(labels))} unique classes.")

    report_df = detect_multimodal_classes(
        embeddings_norm,
        labels,
        min_samples=args.min_samples,
        max_eval_samples=args.max_eval_samples
    )

    print("\n--- Top 10 Most Multimodal Classes ---")
    print(report_df.head(10).to_string(index=False))

    report_df.to_csv(args.output, index=False)
    print(f"\nFull report saved to -> {args.output}")


if __name__ == "__main__":
    main()

