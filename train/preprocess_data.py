import pandas as pd
import re
import random
import argparse
from faker import Faker

# Central dictionary mapping all multilingual placeholders to a canonical English term
PLACEHOLDER_MAP: dict[str, str] = {
    # People / Names
    'person': 'name', 'persona': 'name', 'nome': 'name', 'nombre': 'name', 'navn': 'name',
    'artist_name': 'artist_name', 'name': 'name',

    # Geography / Locations
    'locatie': 'location', 'ort': 'location', 'location': 'location',
    'country': 'country',

    # Dates / Times
    'datum': 'date', 'date': 'date',

    # Media / Entertainment
    'album_name': 'album_name', 'track_name': 'track_name', 'playlist_name': 'playlist_name',
    'movie': 'movie', 'music_genre': 'genre', 'genre': 'genre', 'theme': 'theme',

    # Numbers / Math
    'faces': 'number', 'number': 'number',
    'lower': 'lower', 'upper': 'upper', 'offset': 'offset',

    # Text / Queries
    'satz': 'sentence', 'sætning': 'sentence', 'sentence': 'sentence',
    'forespørgsel': 'query', 'query': 'query'
}


def process_intent_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    for col in ['label', 'sentence', 'lang']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    print(f"Loaded {len(df)} rows. Processing...")

    # Explicitly remove rows that have empty sentence or label
    initial_count = len(df)
    df = df.dropna(subset=['label', 'sentence'])
    df = df[df['label'].astype(str).str.strip() != '']
    df = df[df['sentence'].astype(str).str.strip() != '']
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with empty sentence or label.")

    # ---------------------------------------------------------
    # REQUIREMENT 1: Merge labels into a single ".intent" suffix
    # ---------------------------------------------------------
    def normalize_label(label: str) -> str:
        base_label = re.sub(r'(\.intent)+$', '', str(label).strip())
        return f"{base_label}.intent"

    df['label'] = df['label'].apply(normalize_label)

    # ---------------------------------------------------------
    # REQUIREMENT 2: Create 'entity' column (Normalized to English)
    # ---------------------------------------------------------
    df['original_sentence'] = df['sentence']

    def extract_and_translate_entities(text: str) -> str | None:
        matches = re.findall(r'\{([^}]+)\}', str(text))
        if not matches:
            return None

        # Map found placeholders to their English canonical version.
        # If it's not in the map, it keeps the original name as a fallback.
        english_entities = [PLACEHOLDER_MAP.get(m.lower().strip(), m.lower().strip()) for m in matches]
        return ','.join([e for e in english_entities if e is not None])

    df['entity'] = df['original_sentence'].apply(extract_and_translate_entities)

    # ---------------------------------------------------------
    # REQUIREMENT 3: Interpolate Faker data based on 'lang'
    # ---------------------------------------------------------
    faker_instances = {}

    def get_faker(lang_code: str) -> Faker:
        formatted_lang = str(lang_code).replace('-', '_')
        if formatted_lang not in faker_instances:
            try:
                faker_instances[formatted_lang] = Faker(formatted_lang)
            except AttributeError:
                print(f"Warning: Locale '{formatted_lang}' not supported. Falling back to 'en_US'.")
                faker_instances[formatted_lang] = Faker('en_US')
        return faker_instances[formatted_lang]

    GENRES = ['rock', 'pop', 'jazz', 'hip hop', 'classical', 'electronic', 'country', 'metal', 'blues', 'indie']

    def fill_placeholders(row) -> str:
        text = str(row['original_sentence'])

        if pd.isna(row['entity']):
            return text

        fake = get_faker(row['lang'])

        def replacer(match):
            # Translate the placeholder to English first, so the logic is unified
            raw_p = match.group(1).lower().strip()
            p = PLACEHOLDER_MAP.get(raw_p, raw_p)

            # --- People / Names / Files ---
            if p in ['name', 'artist_name']:
                label = str(row.get('label', '')).lower()
                # If the intent suggests a file/recording, use a file name instead of a person's name
                if 'recording' in label:
                    ext = random.choice(['wav', 'mp3', 'ogg', 'aac'])
                    return f"{fake.file_name(category='text', extension='txt').split('.')[0]}.{ext}"
                if 'dictation' in label:
                    ext = random.choice(['txt', 'doc', 'docx', 'pdf'])
                    return f"{fake.file_name(category='text', extension='txt').split('.')[0]}.{ext}"
                return fake.name()

            # --- Geography / Locations ---
            elif p == 'location':
                return fake.city()
            elif p == 'country':
                return fake.country()

            # --- Dates / Times ---
            elif p == 'date':
                return fake.date_this_year().strftime("%Y-%m-%d")

            # --- Media / Entertainment ---
            elif p in ['album_name', 'track_name', 'playlist_name', 'movie']:
                return fake.catch_phrase().title()
            elif p == 'genre':
                return random.choice(GENRES)
            elif p == 'theme':
                return fake.word()

            # --- Numbers / Math ---
            elif p == 'number':
                return str(fake.random_int(min=1, max=100))
            elif p == 'lower':
                return str(fake.random_int(min=1, max=49))
            elif p == 'upper':
                return str(fake.random_int(min=50, max=100))
            elif p == 'offset':
                return str(fake.random_int(min=1, max=12))

            # --- Text / Queries ---
            elif p == 'sentence':
                return fake.sentence()
            elif p == 'query':
                return fake.sentence(nb_words=4)[:-1]

            # --- Fallback ---
            else:
                return fake.word()

        return re.sub(r'\{([^}]+)\}', replacer, text)

    df['sentence'] = df.apply(fill_placeholders, axis=1)

    # ---------------------------------------------------------
    # REQUIREMENT 4: Save the cleaned up dataset
    # ---------------------------------------------------------
    cols = ['label', 'lang', 'entity', 'original_sentence', 'sentence']
    remaining_cols = [c for c in df.columns if c not in cols]
    final_df = df[cols + remaining_cols]

    final_df.to_csv(output_path, index=False)
    print(f"Successfully processed and saved to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess intent dataset by normalizing labels and filling placeholders.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("output", help="Path to save the processed CSV file.")

    args = parser.parse_args()
    process_intent_dataset(args.input, args.output)

