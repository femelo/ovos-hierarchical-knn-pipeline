"""Create a diverse, balanced subset of the merged intent dataset using LLM Embeddings.

This script selects the **most semantically diverse** K sentences per label
using a greedy farthest-point (maximin) algorithm operating directly on the
pre-computed LLM embedding space.

Outputs
-------
``diverse_subset.csv``      – compact format (lang, label, sentence)
``diverse_subset_full.csv`` – expanded format (lang, domain, intent, sentence)
                              written only when ``merged_intents_dataset_full.csv``
                              is present alongside the input.

A per-label stats CSV (``diverse_subset_stats.csv``) reports how many
sentences were removed and how average semantic diversity changed.
"""

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input is now the Parquet file containing pre-computed embeddings
PARQUET_PATH = os.path.join(OUTPUT_DIR, "dataset/encoded_intents_dataset_granite_97m.parquet")
FULL_CSV_PATH = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")

OUT_PATH = os.path.join(OUTPUT_DIR, "diverse_subset.csv")
OUT_FULL_PATH = os.path.join(OUTPUT_DIR, "diverse_subset_full.csv")
OUT_STATS_PATH = os.path.join(OUTPUT_DIR, "diverse_subset_stats.csv")
OUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "diverse_plots")

# Target number of samples kept per label after diversification for standard intents.
DEFAULT_MAX_PER_LABEL: int = 100

# Data-driven overrides based on LLM Topology Analysis
MACRO_INTENT_CAPS: dict[str, int] = {
    # Tier 1: The Sprawling Galaxies
    "common_query:common_query.intent": 1000,
    "ocp:play.intent": 1000,
    "ovos-skill-wordnet.openvoiceos:definition.intent": 800,
    "ovos-skill-wallpapers.openvoiceos:picture.about.intent": 800,
    "ovos-skill-wallpapers.openvoiceos:wallpaper.about.intent": 800,
    "ovos-skill-weather.openvoiceos:hourly_forecast.intent": 800,

    # Tier 2: The Multi-Modal Archipelagos
    "ovos-skill-moviemaster.openvoiceos:movie.description.intent": 500,
    "ovos-skill-moviemaster.openvoiceos:movie.information.intent": 500,
    "ovos-skill-alerts.openvoiceos:missed_alerts.intent": 500,
    "ovos-skill-dictation.openvoiceos:start_dictation.intent": 400,
    "ovos-skill-dictation.openvoiceos:stop_dictation.intent": 400,
    "ovos-skill-wikipedia.openvoiceos:wiki.intent": 400,
    "ovos-skill-wordnet.openvoiceos:hyponym.intent": 300,
    "ovos-skill-wikihow.openvoiceos:wikihow.intent": 300,
    "ovos-skill-ddg.openvoiceos:search_duck.intent": 300,

    # Tier 3: Medium Sprawl / Noise
    "ovos-skill-weather.openvoiceos:daily_forecast.intent": 250,
    "ovos-skill-date-time.openvoiceos:what.time.is.it.intent": 250,
    "ovos-skill-date-time.openvoiceos:what.time.will.it.be.intent": 250,
    "ovos-skill-news.openvoiceos:news.intent": 250,
    "ovos-skill-icanhazdadjokes.openvoiceos:joke.intent": 200,
}

# Labels with this many samples or fewer are never filtered.
MIN_PER_LABEL: int = 5

# Maximum pool size fed into the distance matrix for very large labels.
MAX_CANDIDATES: int = 10000

# Number of random pairs sampled to estimate average semantic diversity
DIVERSITY_SAMPLE_N: int = 2000

RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Distance / diversity helpers
# ---------------------------------------------------------------------------

def avg_pairwise_cosine_distance(embeddings: np.ndarray, sample_n: int = DIVERSITY_SAMPLE_N) -> float:
    """Estimate mean pairwise cosine distance on a random sample.

    Higher values indicate greater semantic diversity across the LLM embedding space.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n, min(sample_n, n), replace=False)
    sampled = embeddings[idx]

    dist = cosine_distances(sampled)

    # Extract just the upper triangle (exclude diagonal and duplicates)
    upper_tri_indices = np.triu_indices_from(dist, k=1)
    mean_dist = np.mean(dist[upper_tri_indices])

    return float(mean_dist)


# ---------------------------------------------------------------------------
# Core selection algorithm
# ---------------------------------------------------------------------------

def greedy_maximin_emb(embeddings: np.ndarray, k: int) -> list[int]:
    """Select k maximally diverse vectors using greedy farthest-point selection.

    Algorithm operates directly on the semantic LLM embeddings.
    """
    n = len(embeddings)
    if n <= k:
        return list(range(n))

    dist = cosine_distances(embeddings)  # (n, n) float32

    # Seed: embedding with the highest average distance to all others
    seed = int(dist.mean(axis=1).argmax())
    selected = [seed]

    remaining = np.ones(n, dtype=bool)
    remaining[seed] = False

    min_dist = dist[seed].copy()

    for _ in range(k - 1):
        scores = np.where(remaining, min_dist, -1.0)
        next_idx = int(scores.argmax())
        selected.append(next_idx)
        remaining[next_idx] = False
        np.minimum(min_dist, dist[next_idx], out=min_dist)

    return selected


def diverse_select_emb(sentences: list[str], embeddings: np.ndarray, k: int) -> list[int]:
    """Full diverse-selection pipeline returning indices into ``sentences``/``embeddings``."""

    # -- Step 1: word-bag deduplication -------------------------------------
    # We still use this to quickly prune exact structural duplicates before matrix math
    bag_to_best: dict[frozenset, tuple[int, int]] = {}
    for i, s in enumerate(sentences):
        bag = frozenset(s.split())
        length = len(s)
        if bag not in bag_to_best or length > bag_to_best[bag][0]:
            bag_to_best[bag] = (length, i)

    dedup_indices = [idx for _, idx in bag_to_best.values()]

    if len(dedup_indices) <= k:
        return dedup_indices

    # -- Step 2: pre-sample if pool is still very large ---------------------
    rng = np.random.default_rng(RANDOM_STATE)
    if len(dedup_indices) > MAX_CANDIDATES:
        pre = rng.choice(len(dedup_indices), MAX_CANDIDATES, replace=False)
        pool_indices = [dedup_indices[i] for i in pre]
    else:
        pool_indices = dedup_indices

    # -- Step 3: greedy maximin on the semantic space ----------------------
    pool_embeddings = embeddings[pool_indices]
    picked_in_pool = greedy_maximin_emb(pool_embeddings, k)

    return [pool_indices[i] for i in picked_in_pool]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_diversification(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    stats_df: pd.DataFrame,
    plots_dir: str = OUT_PLOTS_DIR,
) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    counts_before = df_before["label"].value_counts()
    counts_after  = df_after["label"].value_counts()

    # 1. Label size distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(
        np.log10(max(1, min(counts_before.min(), counts_after.min()))),
        np.log10(counts_before.max()),
        40,
    )
    ax.hist(counts_before.values, bins=bins, alpha=0.6, label="Before", color="steelblue")
    ax.hist(counts_after.values,  bins=bins, alpha=0.6, label="After",  color="darkorange")
    ax.set_xscale("log")
    ax.set_xlabel("Examples per label (log scale)")
    ax.set_ylabel("Number of labels")
    ax.set_title(
        f"Label size distribution  "
        f"(before: {len(df_before):,}  after: {len(df_after):,} examples)"
    )
    ax.axvline(DEFAULT_MAX_PER_LABEL, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Default Cap ({DEFAULT_MAX_PER_LABEL})")

    unique_macro_caps = sorted(set(MACRO_INTENT_CAPS.values()))
    for cap in unique_macro_caps:
        ax.axvline(cap, color="tomato", linestyle=":", linewidth=1.0, alpha=0.8)
    if unique_macro_caps:
        ax.plot([], [], color="tomato", linestyle=":", linewidth=1.0, alpha=0.8, label="Macro Caps")

    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "label_size_distribution.png"), dpi=120)
    plt.close(fig)

    # 2. Kept-% distribution
    trimmed = stats_df[stats_df["n_before"] > MIN_PER_LABEL]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(trimmed["kept_pct"], bins=25, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(trimmed["kept_pct"].mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean = {trimmed['kept_pct'].mean():.1f}%")
    ax.axvline(100, color="grey", linestyle=":", linewidth=0.8, label="100% (no trim)")
    ax.set_xlabel("% of examples kept per label")
    ax.set_ylabel("Number of labels")
    ax.set_title("Distribution of kept fraction across labels")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "kept_pct_distribution.png"), dpi=120)
    plt.close(fig)

    # 3. Diversity scatter (Now based on LLM Semantic Distance)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        stats_df["div_before"], stats_df["div_after"],
        c=stats_df["kept_pct"], cmap="RdYlGn", alpha=0.6,
        s=18, vmin=0, vmax=100,
    )
    lim_max = max(stats_df["div_before"].max(), stats_df["div_after"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, label="No change")
    plt.colorbar(sc, ax=ax, label="% examples kept")
    ax.set_xlabel("Semantic Cosine Distance (Before)")
    ax.set_ylabel("Semantic Cosine Distance (After)")
    ax.set_title("Semantic Diversity Change per Label\n(above diagonal = improved)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "diversity_scatter.png"), dpi=120)
    plt.close(fig)

    # 4. Diversity gain histogram
    gain = stats_df["div_after"] - stats_df["div_before"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(gain, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0,          color="grey",  linestyle=":",  linewidth=0.8, label="No change")
    ax.axvline(gain.mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean gain = {gain.mean():+.4f}")
    ax.set_xlabel("Semantic Diversity Gain  (div_after − div_before)")
    ax.set_ylabel("Number of labels")
    ax.set_title("Distribution of Semantic Diversity Improvement")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "diversity_gain_hist.png"), dpi=120)
    plt.close(fig)

    # 5. Language balance
    lang_before = df_before["lang"].value_counts().rename("Before")
    lang_after  = df_after["lang"].value_counts().rename("After")
    lang_cmp = pd.concat([lang_before, lang_after], axis=1).fillna(0).astype(int)
    lang_cmp = lang_cmp.sort_values("Before", ascending=False)

    x = np.arange(len(lang_cmp))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(lang_cmp) * 1.1), 5))
    bars_b = ax.bar(x - width / 2, lang_cmp["Before"], width, label="Before", color="steelblue")
    bars_a = ax.bar(x + width / 2, lang_cmp["After"],  width, label="After",  color="darkorange")
    for xpos, (_, row) in zip(x, lang_cmp.iterrows()):
        pct = 100 * row["After"] / row["Before"] if row["Before"] > 0 else 0
        ax.text(xpos, max(row["Before"], row["After"]) * 1.01,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color="dimgrey")
    ax.set_xticks(x)
    ax.set_xticklabels(lang_cmp.index, fontsize=9)
    ax.set_ylabel("Examples")
    ax.set_title("Language distribution before vs after diversification")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_balance.png"), dpi=120)
    plt.close(fig)

    # 6. Deduplication pipeline
    top_n = 30
    reduced = (
        stats_df[stats_df["n_before"] > stats_df["n_after"]]
        .nlargest(top_n, "n_before")
        .sort_values("n_before", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(14, max(6, len(reduced) * 0.42)))
    y = np.arange(len(reduced))
    ax.barh(y, reduced["n_before"],  height=0.7, label="Original",   color="steelblue",  alpha=0.9)
    ax.barh(y, reduced["n_deduped"], height=0.7, label="After dedup", color="darkorange", alpha=0.85)
    ax.barh(y, reduced["n_after"],   height=0.7, label="Final",       color="seagreen",   alpha=0.9)

    target_caps = [MACRO_INTENT_CAPS.get(lbl, DEFAULT_MAX_PER_LABEL) for lbl in reduced["label"]]
    ax.scatter(target_caps, y, color="black", marker="|", s=100, zorder=5, label="Target Cap")

    ax.set_yticks(y)
    ax.set_yticklabels(reduced["label"], fontsize=7)
    ax.set_xlabel("Example count")
    ax.set_title(f"Top {len(reduced)} most-reduced labels: three-stage pipeline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "dedup_pipeline.png"), dpi=120)
    plt.close(fig)

    logger.info(f"Saved diversification plots → {plots_dir}/  (6 files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    before = len(df)

    # Drop rows with missing text or labels
    df = df.dropna(subset=["sentence", "label"])
    df["sentence"] = df["sentence"].astype(str)

    if len(df) < before:
        logger.warning(f"  Dropped {before - len(df):,} rows with missing sentences")
    logger.info(f"  {len(df):,} rows  |  {df['label'].nunique()} labels")

    # Dynamically extract embedding columns
    emb_cols = [col for col in df.columns if col.startswith("dim_")]
    if not emb_cols:
        raise ValueError("No embedding columns found in Parquet. Must be named 'dim_0', 'dim_1', etc.")
    logger.info(f"  Detected {len(emb_cols)} embedding dimensions.")

    full_df: pd.DataFrame | None = None
    if os.path.exists(FULL_CSV_PATH):
        full_df = pd.read_csv(FULL_CSV_PATH)
        logger.info(f"  Full dataset loaded ({len(full_df):,} rows)")

    selected_rows: list[pd.DataFrame] = []
    stats_rows: list[dict] = []

    labels = df["label"].unique()
    logger.info(
        f"Selecting semantically diverse sentences per label using dynamic caps "
        f"(Default: {DEFAULT_MAX_PER_LABEL}, Min kept intact: {MIN_PER_LABEL}) ..."
    )

    for label in tqdm(labels, desc="Labels", unit="label"):
        # Create a mask to slice both the dataframe and the embeddings safely
        mask = (df["label"] == label)
        group = df[mask]

        sentences = group["sentence"].tolist()
        # Extract and cast embeddings to float32 to save RAM
        embeddings = df.loc[mask, emb_cols].values.astype(np.float32)
        n_before = len(sentences)

        if n_before <= MIN_PER_LABEL:
            selected_rows.append(group)
            stats_rows.append({
                "label": label,
                "n_before": n_before,
                "n_after": n_before,
                "n_deduped": n_before,
                "div_before": round(avg_pairwise_cosine_distance(embeddings), 4),
                "div_after": round(avg_pairwise_cosine_distance(embeddings), 4),
                "kept_pct": 100.0,
            })
            continue

        # DYNAMIC CAP ASSIGNMENT
        target_k = MACRO_INTENT_CAPS.get(label, DEFAULT_MAX_PER_LABEL)
        k = min(target_k, n_before)

        div_before = avg_pairwise_cosine_distance(embeddings)

        # Pass both sentences AND embeddings to the selection logic
        chosen_indices = diverse_select_emb(sentences, embeddings, k)

        chosen_df = group.iloc[chosen_indices]
        chosen_embeddings = embeddings[chosen_indices]

        div_after = avg_pairwise_cosine_distance(chosen_embeddings)
        n_deduped = len({frozenset(s.split()) for s in sentences})

        stats_rows.append({
            "label": label,
            "n_before": n_before,
            "n_deduped": n_deduped,
            "n_after": len(chosen_df),
            "div_before": round(div_before, 4),
            "div_after": round(div_after, 4),
            "kept_pct": round(100 * len(chosen_df) / n_before, 1),
        })

        selected_rows.append(chosen_df)

    # Assemble and save outputs
    subset_df = pd.concat(selected_rows, ignore_index=True)

    # Strip the heavy embedding columns before saving the CSV to keep it lightweight
    keep_cols = ["lang", "label", "sentence"]
    subset_df[keep_cols].to_csv(OUT_PATH, index=False)
    logger.info(f"Saved compact subset → {OUT_PATH}  ({len(subset_df):,} rows)")

    if full_df is not None:
        merge_keys = ["lang", "label", "sentence"]
        full_subset = subset_df[keep_cols].merge(
            full_df[["lang", "domain", "intent", "sentence"]],
            on=["lang", "sentence"],
            how="left",
        ).drop_duplicates(subset=merge_keys)
        full_subset[["lang", "domain", "intent", "sentence"]].to_csv(OUT_FULL_PATH, index=False)
        logger.info(f"Saved full subset    → {OUT_FULL_PATH}  ({len(full_subset):,} rows)")

    stats_df = pd.DataFrame(stats_rows).sort_values("n_before", ascending=False)
    stats_df.to_csv(OUT_STATS_PATH, index=False)
    logger.info(f"Saved stats          → {OUT_STATS_PATH}")

    n_orig = len(df)
    n_new = len(subset_df)
    logger.info("=" * 60)
    logger.info(f"Total examples : {n_orig:>8,}  →  {n_new:>8,}  ({100*n_new/n_orig:.1f}%)")
    logger.info(f"Labels         : {len(labels):>8,}")
    logger.info(
        f"Avg diversity  : {stats_df['div_before'].mean():.4f}  →  "
        f"{stats_df['div_after'].mean():.4f}  (LLM Semantic Cosine Distance)"
    )

    trimmed = stats_df[stats_df["n_before"] > stats_df["n_after"]].head(10)
    if not trimmed.empty:
        logger.info("\nTop 10 most trimmed labels:")
        for _, row in trimmed.iterrows():
            logger.info(
                f"  {row['label']:<60}  "
                f"{row['n_before']:>5} → {row['n_after']:>3}  "
                f"div {row['div_before']:.3f} → {row['div_after']:.3f}"
            )

    plot_diversification(df, subset_df, stats_df)


if __name__ == "__main__":
    main()

