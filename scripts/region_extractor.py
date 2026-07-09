"""
region_extractor.py
--------------------
Builds two embedding lookup tables from the fine-tuned triplet model:

  - region_embeddings_lookup: one embedding per canonical GLOBAL_REGION (9 rows)
  - focus_embeddings_lookup:  one embedding per curated imaging focus, labeled
                               with its mapped region (from ontology_mapper's
                               build_focus_to_region_map)

These tables are the basis for extracting region/focus from free-text exam
descriptions by nearest-neighbor lookup against the embedding space.

Usage:
    python3 scripts/region_extractor.py
    python3 scripts/region_extractor.py --model output/all_minilm_stratified --out-dir output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ontology_mapper import GLOBAL_REGIONS, build_focus_to_region_map, MODALITY_ONTOLOGY

# Modality abbreviations that show up as words in study text but are never
# anatomical terms — must not be allowed to win the KNN lookup.
MODALITY_CODES = set(MODALITY_ONTOLOGY.keys()) | {
    'MRI', 'CTA', 'MRA', 'XRAY', 'PET', 'DXA', 'SPECT', 'US', 'RF',
}

DEFAULT_MIN_SIMILARITY = 0.75


def load_model(model_path):
    """Load a SentenceTransformer model (fine-tuned or base)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_path)


def build_region_embeddings_lookup(model):
    """Embed each of the 9 canonical global regions."""
    regions = GLOBAL_REGIONS
    embeddings = model.encode(regions, convert_to_numpy=True)
    return pd.DataFrame({
        'region': regions,
        'embedding': [json.dumps(e.tolist()) for e in embeddings],
    })


def build_focus_embeddings_lookup(model):
    """Embed every curated focus from ontology_mapper's focus→region map."""
    focus_to_region = build_focus_to_region_map()

    foci, regions = [], []
    for region, focus_list in focus_to_region.items():
        for focus in focus_list:
            foci.append(focus)
            regions.append(region)

    embeddings = model.encode(foci, convert_to_numpy=True)
    return pd.DataFrame({
        'focus': foci,
        'region': regions,
        'embedding': [json.dumps(e.tolist()) for e in embeddings],
    })


def decode_embedding_column(series):
    """Decode a column of JSON-string embeddings (as written to the lookup CSVs) into an ndarray."""
    return np.array([json.loads(e) if isinstance(e, str) else e for e in series])


def load_embedding_table(csv_path):
    """Read back a lookup table, decoding the embedding column into an ndarray."""
    df = pd.read_csv(csv_path)
    embeddings = decode_embedding_column(df['embedding'])
    return df, embeddings


def tokenize_words(text):
    """Split free text into individual alphabetic words (drops numbers/punctuation)."""
    return re.findall(r"[A-Za-z]+", text)


def cosine_sim_matrix(a, b):
    """Row-normalized cosine similarity between every row of a and every row of b."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _best_word_match(text, model, candidate_embeddings, min_similarity):
    """
    Embed each word in `text` individually and return the (word_idx, candidate_idx,
    similarity) triple closest to ANY single word — avoids the noise of embedding
    the whole string (modality codes, view counts, etc. drag a whole-string
    embedding away from the anatomical term it actually contains).

    Known modality abbreviations (CT/MR/US/...) are stripped before scoring so
    they can't win purely on being short, common tokens. Returns None if no
    word clears `min_similarity` — a low score usually means the term wasn't
    covered by training data rather than a real match.
    """
    words = tokenize_words(text)
    filtered = [w for w in words if w.upper() not in MODALITY_CODES]
    if filtered:
        words = filtered
    if not words:
        return None

    word_embeddings = model.encode(words, convert_to_numpy=True)
    sims = cosine_sim_matrix(word_embeddings, candidate_embeddings)  # (n_words, n_candidates)
    word_idx, candidate_idx = np.unravel_index(np.argmax(sims), sims.shape)
    similarity = float(sims[word_idx, candidate_idx])

    if similarity < min_similarity:
        return None

    return words[word_idx], candidate_idx, similarity


def extract_region(text, model, region_names, region_embeddings, min_similarity=DEFAULT_MIN_SIMILARITY):
    """Return the region whose embedding is closest to any single word in `text`."""
    match = _best_word_match(text, model, region_embeddings, min_similarity)
    if match is None:
        return None
    word, region_idx, similarity = match
    return {
        'region': region_names[region_idx],
        'matched_word': word,
        'similarity': similarity,
    }


def extract_focus(text, model, focus_names, focus_regions, focus_embeddings, min_similarity=DEFAULT_MIN_SIMILARITY):
    """Return the imaging focus (and its mapped region) whose embedding is closest to any single word in `text`."""
    match = _best_word_match(text, model, focus_embeddings, min_similarity)
    if match is None:
        return None
    word, focus_idx, similarity = match
    return {
        'focus': focus_names[focus_idx],
        'region': focus_regions[focus_idx],
        'matched_word': word,
        'similarity': similarity,
    }


def main():
    parser = argparse.ArgumentParser(description='Build region and focus embedding lookup tables.')
    parser.add_argument('--model', default=None, help='Path to SentenceTransformer model (default: output/all_minilm_stratified)')
    parser.add_argument('--out-dir', default=None, help='Output directory (default: output/)')
    parser.add_argument('--query', default=None, help='Test region extraction on a piece of free text')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    model_path = args.model or str(script_dir / 'output' / 'all_minilm_stratified')
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    print("Embedding regions...")
    region_df = build_region_embeddings_lookup(model)
    region_path = out_dir / 'region_embeddings_lookup.csv'
    region_df.to_csv(region_path, index=False)
    print(f"  {len(region_df)} regions -> {region_path}")

    print("Embedding foci...")
    focus_df = build_focus_embeddings_lookup(model)
    focus_path = out_dir / 'focus_embeddings_lookup.csv'
    focus_df.to_csv(focus_path, index=False)
    print(f"  {len(focus_df)} foci -> {focus_path}")

    if args.query:
        region_embeddings = decode_embedding_column(region_df['embedding'])
        focus_embeddings = decode_embedding_column(focus_df['embedding'])

        region_result = extract_region(args.query, model, region_df['region'].tolist(), region_embeddings)
        focus_result = extract_focus(args.query, model, focus_df['focus'].tolist(), focus_df['region'].tolist(), focus_embeddings)

        print(f"\nQuery: {args.query!r}")
        if region_result:
            print(f"  region -> {region_result['region']} (matched word: {region_result['matched_word']!r}, similarity: {region_result['similarity']:.4f})")
        else:
            print("  region -> no confident match")
        if focus_result:
            print(f"  focus  -> {focus_result['focus']} [{focus_result['region']}] (matched word: {focus_result['matched_word']!r}, similarity: {focus_result['similarity']:.4f})")
        else:
            print("  focus  -> no confident match")


if __name__ == '__main__':
    main()
