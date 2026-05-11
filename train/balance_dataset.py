import pandas as pd
import numpy as np
import argparse
import os
import re
import random
import time
import faiss
import google.generativeai as genai  # type: ignore
from sklearn.cluster import MiniBatchKMeans
from ovos_hierarchical_knn_pipeline.encoders import load_encoder
from faker import Faker

# Central placeholder map
PLACEHOLDER_MAP = {
    "person": "name",
    "persona": "name",
    "nome": "name",
    "nombre": "name",
    "navn": "name",
    "artist_name": "artist_name",
    "name": "name",
    "locatie": "location",
    "ort": "location",
    "location": "location",
    "country": "country",
    "datum": "date",
    "date": "date",
    "album_name": "album_name",
    "track_name": "track_name",
    "playlist_name": "playlist_name",
    "movie": "movie",
    "music_genre": "genre",
    "genre": "genre",
    "theme": "theme",
    "faces": "number",
    "number": "number",
    "lower": "lower",
    "upper": "upper",
    "offset": "offset",
    "satz": "sentence",
    "sætning": "sentence",
    "sentence": "sentence",
    "forespørgsel": "query",
    "query": "query",
}
GENRES = [
    "rock",
    "pop",
    "jazz",
    "hip hop",
    "classical",
    "electronic",
    "country",
    "metal",
    "blues",
    "indie",
]


def get_faker(lang_code, cache):
    formatted_lang = str(lang_code).replace("-", "_")
    if formatted_lang not in cache:
        try:
            cache[formatted_lang] = Faker(formatted_lang)
        except AttributeError:
            cache[formatted_lang] = Faker("en_US")
    return cache[formatted_lang]


def generate_faker_variants(row, num_variants, faker_cache):
    """Generates N semantic variants using Faker for a row with placeholders."""
    if pd.isna(row.get("entity")) or not row.get("entity"):
        return []

    fake = get_faker(row["lang"], faker_cache)
    variants = []

    for _ in range(num_variants):

        def replacer(match):
            raw_p = match.group(1).lower().strip()
            p = PLACEHOLDER_MAP.get(raw_p, raw_p)

            if p in ["name", "artist_name"]:
                label = str(row.get("label", "")).lower()
                if "recording" in label:
                    ext = random.choice(["wav", "mp3", "ogg", "aac"])
                    return f"{fake.file_name(category='text', extension='txt').split('.')[0]}.{ext}"
                if "dictation" in label:
                    ext = random.choice(["txt", "doc", "docx", "pdf"])
                    return f"{fake.file_name(category='text', extension='txt').split('.')[0]}.{ext}"
                return fake.name()
            elif p == "location":
                return fake.city()
            elif p == "country":
                return fake.country()
            elif p == "date":
                return fake.date_this_year().strftime("%Y-%m-%d")
            elif p in ["album_name", "track_name", "playlist_name", "movie"]:
                return fake.catch_phrase().title()
            elif p == "genre":
                return random.choice(GENRES)
            elif p == "theme":
                return fake.word()
            elif p == "number":
                return str(fake.random_int(min=1, max=100))
            elif p == "lower":
                return str(fake.random_int(min=1, max=49))
            elif p == "upper":
                return str(fake.random_int(min=50, max=100))
            elif p == "offset":
                return str(fake.random_int(min=1, max=12))
            elif p == "sentence":
                return fake.sentence()
            elif p == "query":
                return fake.sentence(nb_words=4)[:-1]
            else:
                return fake.word()

        new_sentence = re.sub(r"\{([^}]+)\}", replacer, str(row["original_sentence"]))

        new_row = row.to_dict()
        new_row["sentence"] = new_sentence
        new_row["embedding"] = None
        variants.append(new_row)

    return variants


def generate_llm_paraphrases(examples, lang, needed):
    """Calls Gemini-3.1-Flash-Lite to generate paraphrases."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY missing. Skipping LLM augmentation.")
        return []

    genai.configure(api_key=api_key)  # type: ignore

    # Initialize the specific Gemini model
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")  # type: ignore

    prompt = (
        f"You are a linguistic expert. I need exactly {needed} new, semantically equivalent "
        f"sentences in the language '{lang}' based on these examples:\n"
        f"{chr(10).join(examples)}\n\n"
        f"Respond ONLY with the generated sentences, one per line, with no numbers, formatting, bullets, or markdown."
    )

    try:
        # Sleep for 4 seconds to respect the 15 RPM free tier limit
        time.sleep(4)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(  # type: ignore
                temperature=0.7,
            ),
        )

        # Clean up the output and split into list
        content = response.text
        new_sentences = [line.strip() for line in content.split("\n") if line.strip()]

        # Strip any accidental leading hyphens or numbers the LLM might have added
        new_sentences = [re.sub(r"^[\d\-\.\*]+\s*", "", s) for s in new_sentences]

        return new_sentences[:needed]
    except Exception as e:
        print(f"Gemini API Call failed: {e}")
        return []


def main(input_path, output_path, model_name, encoder_file="model.onnx"):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows. Encoding sentences with {model_name}...")

    encoder = load_encoder(model_name, onnx_filename=encoder_file)
    embeddings = encoder.encode_documents(df["sentence"].tolist(), show_progress_bar=True)
    df["embedding"] = list(embeddings)

    balanced_rows = []
    faker_cache = {}

    class_counts = df["label"].value_counts()

    for label, count in class_counts.items():
        class_mask = df["label"] == label
        class_df = df[class_mask]

        # -------------------------------------------------
        # TIER 1: The Goliaths (>= 1,000,000) -> Cap at 5,000
        # -------------------------------------------------
        if count >= 1_000_000:
            print(f"Clustering Goliath class '{label}' ({count} -> 5,000)...")
            X = embeddings[class_mask.values]

            kmeans = MiniBatchKMeans(
                n_clusters=5000, batch_size=10240, random_state=42, n_init="auto"
            )
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_.astype(np.float32)  # type: ignore
            faiss.normalize_L2(centroids)

            index = faiss.IndexFlatIP(X.shape[1])
            index.add(X)  # type: ignore
            _, I_ = index.search(centroids, 1)  # type: ignore
            closest_indices = I_.flatten()

            # Use the centroids as the embedding for these rows
            sampled_df = class_df.iloc[closest_indices].copy()
            sampled_df["embedding"] = list(centroids)
            balanced_rows.append(sampled_df)

        # -------------------------------------------------
        # TIER 2: The Mid-Tier (5,000 to 1,000,000) -> Cap at 5,000
        # -------------------------------------------------
        elif count > 5000:
            print(f"Downsampling Mid-tier class '{label}' ({count} -> 5,000)...")
            sampled = class_df.groupby("lang", group_keys=False).apply(
                lambda x: x.sample(int(np.ceil(len(x) * 5000 / count)), random_state=42)
            )
            if len(sampled) > 5000:
                sampled = sampled.sample(5000, random_state=42)
            balanced_rows.append(sampled)

        # -------------------------------------------------
        # TIER 3: The Micro-Tier (< 100) -> Augment to 100
        # -------------------------------------------------
        elif count < 100:
            print(f"Augmenting Micro-tier class '{label}' ({count} -> 100)...")
            current_df = class_df.copy()
            augmented_data = []

            # Step A: Faker Expansion
            for _, row in current_df.iterrows():
                if pd.notna(row.get("entity")):  # type: ignore
                    variants = generate_faker_variants(
                        row, num_variants=10, faker_cache=faker_cache
                    )
                    augmented_data.extend(variants)

            if augmented_data:
                aug_df = pd.DataFrame(augmented_data)
                current_df = pd.concat([current_df, aug_df], ignore_index=True)
                assert isinstance(current_df, pd.DataFrame)

            # Step B: LLM Expansion if still under 100
            current_count = len(current_df)
            if current_count < 100:
                needed = 100 - current_count

                for lang in current_df["lang"].unique():  # type: ignore
                    lang_examples = (
                        current_df[current_df["lang"] == lang]["sentence"]  # type: ignore
                        .dropna()  # type: ignore
                        .unique()
                        .tolist()
                    )
                    if not lang_examples:
                        continue

                    sample_examples = lang_examples[:5]
                    new_sentences = generate_llm_paraphrases(
                        sample_examples, lang, needed
                    )

                    for sent in new_sentences:
                        new_row = current_df.iloc[0].to_dict()
                        new_row["sentence"] = sent
                        new_row["original_sentence"] = sent
                        new_row["entity"] = None
                        new_row["embedding"] = None
                        augmented_data.append(new_row)

                if augmented_data:
                    current_df = pd.DataFrame(
                        current_df.to_dict("records") + augmented_data  # type: ignore
                    ).drop_duplicates("sentence")

            balanced_rows.append(current_df)

        # -------------------------------------------------
        # TIER 4: The Normal Tier (100 to 5,000) -> Keep As Is
        # -------------------------------------------------
        else:
            balanced_rows.append(class_df)

    final_df = pd.concat(balanced_rows, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Identify rows that need new embeddings (the augmented ones)
    missing_mask = final_df["embedding"].isna()
    if missing_mask.any():  # type: ignore
        print("Computing embeddings for newly augmented sentences...")
        missing_sentences = final_df.loc[missing_mask, "sentence"].tolist()
        new_embeddings = encoder.encode_documents(missing_sentences, show_progress_bar=True)
        # Re-assign back to dataframe
        final_df.loc[missing_mask, "embedding"] = pd.Series(list(new_embeddings), index=final_df[missing_mask].index)

    # Convert the "embedding" column (list of floats) into individual columns (dim_0, dim_1, ...)
    print("Expanding embeddings into separate columns...")
    emb_matrix = np.vstack(final_df["embedding"].values)  # type: ignore
    emb_df = pd.DataFrame(emb_matrix, columns=["dim_" + str(i) for i in range(emb_matrix.shape[1])])  # type: ignore

    # Drop the original embedding list column and concatenate the new dimension columns
    final_df = final_df.drop(columns=["embedding"])
    final_df = pd.concat([final_df, emb_df], axis=1)

    # Save as parquet, ensuring path ends in .parquet
    if output_path.endswith(".csv"):
        output_path = output_path.rsplit(".", 1)[0] + ".parquet"

    final_df.to_parquet(output_path, index=False)

    print(f"\nPipeline complete! Balanced dataset saved to {output_path}.")
    print(f"Original size: {len(df)} | New size: {len(final_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balance intent dataset using FAISS KMeans and Gemini API."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to cleaned dataset CSV"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save balanced dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/granite-97m-onnx",
        help="Model path. Auto-detected: EmbeddingGemma ONNX if <path>/onnx/model_q4.onnx exists, "
             "Granite ONNX if <path>/onnx/model_quint8_avx2.onnx exists, "
             "otherwise model2vec StaticModel.",
    )
    parser.add_argument(
        "--encoder-file",
        type=str,
        default="model.onnx",
        help="ONNX filename inside <model>/onnx/ to load "
             "(default: 'model.onnx' for full F32 precision). "
             "Use 'model_quint8_avx2.onnx' for the quantised AVX2 variant.",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.encoder_file)

