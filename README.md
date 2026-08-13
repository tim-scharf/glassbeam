# hf-intel

Classifies raw radiology study descriptions (the free-text `study_desc_raw`
strings scanners/EHRs produce, e.g. `"CT CHEST ABDOMEN PELVIS W AND WO
CONTRAST"`) into structured attributes: **region/focus**, **laterality**,
**contrast timing**, and **technique/study type**.

## How it works

**The rules were built offline with LLM assistance, not learned at
runtime.** An LLM was used to comb through the ~6,100 unique
`study_desc_raw` values in the training corpus (`output/glassbeam_data.csv`,
18 source customers) and work out the keyword patterns, typo variants, and
edge cases that separate one category from another for each attribute. The
result of that offline analysis is a set of deterministic, human-readable
keyword-cascade rules — plain Python `if`/`elif` chains and lookup
dictionaries in `scripts/*_extractor.py` — not a model checkpoint. **There
is no LLM call, embedding, or network request at classification time**:
every prediction the pipeline makes is a fast, explainable, fully
reproducible string match. This mirrors how the pipeline was actually
developed — each extractor's docstring documents the specific rule order
and edge cases that were worked out against the real corpus (typos like
"CONRAST"/"CONTAST", the `"L SPINE"` vs. left-side ambiguity, `"BILAT AP
STANDING AND LAT LEFT"` still being a bilateral study, etc.).

### Pipeline

```
raw CSV (Larger_Dataset/*.xlsx)
        │  parse_study_data.py
        ▼
output/glassbeam_data.csv   (modality, study_desc_raw, ...)
        │
        │  classify.py  ── routes each (modality, study_desc_raw) pair
        │                   through data/modality_model_architecture.json
        ▼
   ┌────────────┬─────────────┬──────────┬────────────────────────┐
   │ region/    │ laterality  │ contrast │ technique/study_type   │
   │ focus      │             │ timing   │                        │
   └────────────┴─────────────┴──────────┴────────────────────────┘
        │
        ▼
data/classified_studies.json  (predictions + per-attribute confidence)
```

1. **`parse_study_data.py`** reads the raw Excel exports in
   `data/Larger_Dataset/`, combines them, and writes a single
   `output/glassbeam_data.csv`.
2. **`data/modality_model_architecture.json`** is a routing table saying
   which attributes even apply to a given modality — e.g. mammography
   (`MG`) has no meaningful "region" (it's always breast) so it only gets
   laterality and technique/study type, while nuclear medicine (`NM`) has
   no laterality concept at all. This keeps "not applicable" (`None`)
   distinct from "the model ran and found nothing" (e.g. `"Unspecified"`).
3. **`classify.py`** is the single entry point. For each `(modality,
   study_desc_raw)` pair it consults the routing table once, then calls
   into whichever of the four extractors apply:
   - `region_focus_extractor.py` — multi-label anatomical region (Chest,
     Abdomen, Head, ...) and, within each region, a specific focus
     (liver, kidney, brain, ...).
   - `laterality_extractor.py` — Right / Left / Bilateral / Unspecified.
   - `contrast_extractor.py` — With / Without / With and without /
     Unspecified.
   - `technique_extractor.py` — a fixed category list, only for the two
     modalities (`NM`, `MG`) where technique cleanly partitions the data.
4. **`confidence.py`** scores every prediction from two independent,
   explainable signals — no embeddings here either:
   - **Rule tier**: each extractor reports *which* rule in its cascade
     fired (an explicit keyword match scores higher than a weak
     last-resort heuristic, or a case where two categories' patterns
     collided).
   - **Novelty**: the fraction of the description's words that never
     appeared anywhere in the training corpus. High novelty means the
     rules are operating outside the vocabulary they were built against,
     so their score gets penalized.

   `confidence = tier_base_score - 0.4 * novelty`, clamped to `[0, 1]`.

### Reporting

- `build_architecture_summary.py` and `build_performance_deck.py` generate
  `.pptx` decks summarizing the pipeline architecture and held-out
  accuracy against a genuinely unseen sample.
- `generate_all_modalities.py` builds a browsable HTML reference from the
  LOINC/RSNA Radiology Playbook (`data/LoincRsnaRadiologyPlaybook.csv`).

## Project Structure

```
hf-intel/
├── data/                              # Ontology JSONs, routing table, raw Excel exports
├── output/                            # Combined CSV + classification results
├── scripts/
│   ├── parse_study_data.py            # Excel -> glassbeam_data.csv
│   ├── modality_architecture.py       # Loads the modality routing table
│   ├── region_focus_extractor.py      # Region/focus rules
│   ├── laterality_extractor.py        # Laterality rules
│   ├── contrast_extractor.py          # Contrast-timing rules
│   ├── technique_extractor.py         # Technique/study-type rules
│   ├── confidence.py                  # Tier + novelty confidence scoring
│   ├── classify.py                    # Orchestrates the four extractors
│   ├── build_architecture_summary.py  # Architecture .pptx
│   ├── build_performance_deck.py      # Held-out accuracy .pptx
│   └── generate_all_modalities.py     # LOINC/RSNA reference HTML
├── requirements.txt
└── README.md
```

## Getting Started

1. Set up virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Build the combined dataset from the raw Excel exports:
```bash
python3 scripts/parse_study_data.py
```

Classify a single description:
```bash
python3 scripts/classify.py --modality CT --text "CT CHEST ABDOMEN PELVIS W AND WO CONTRAST"
```

Classify every unique `(modality, study_desc_raw)` pair in the dataset:
```bash
python3 scripts/classify.py --csv output/glassbeam_data.csv --out data/classified_studies.json
```

## Contributing

Each extractor's rules were derived by reviewing the actual corpus, not
guessed in the abstract — if you're adding a new keyword or category,
check it against `output/glassbeam_data.csv` first, and add a line to the
extractor's docstring explaining what real description motivated it.
