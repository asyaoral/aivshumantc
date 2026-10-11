Place the exported main CSV here as `ai_vs_human.csv`.

In Apple Numbers open `AI_vs_Human_Text_Classification_augmented.numbers` → **File → Export To → CSV** → save as `data/raw/ai_vs_human.csv`.

CSV MUST have header `Text,Class` (comma **or** semicolon delimiter). Classes: `0` AI, `1` Human.

The 200-example flipped-label diagnostic belongs to a separate experiment and must never be merged into the primary dataset.

Raw text may contain copyrighted material. Before public GitHub upload, document **actual sources and redistribution rights**. Data files are ignored by default for your safety.
