# Contribution — Asya Oral

| | |
| --- | --- |
| Student ID | 2309011045 |
| GitHub | [@asyaoral](https://github.com/asyaoral) |
| Areas | Data preparation (shared), **coding**, **model training**, **documentation** |

## Summary
I took part in the team's shared data preparation. I **coded** the feature and training pipeline as reusable scripts and as a leakage-free notebook. I worked on **model training** and wrote the **technical documentation** of the pipeline.

## 1. Data preparation (shared with Nilasu Yıldız and Beyza Eser)
- Collected human-written English passages from books, articles and similar published sources (label 1).
- Generated AI texts on comparable topics with several different AI models (label 0).
- Labelled the texts and combined them into one table (`Text`, `Class`), kept in Apple Numbers and exported to CSV.
- Checked class balance and manually looked for empty, very short or duplicated texts.

## 2. Coding
### `src/features.py`
- **`LinguisticFeatures`:** a scikit-learn transformer that computes six stylistic features per text:

| Feature | Definition |
| --- | --- |
| Avg_Sentence_Length | words / sentences |
| Lexical_Diversity | unique words / words |
| Punctuation_Density | `. , ; : ! ?` / words |
| Capitalized_Ratio | capitalized words / words |
| Pronoun_Ratio | *i, we, my, our, us, their* / words |
| Num_Sentences | number of sentences |

- **`make_model()`:** builds every experiment as one `Pipeline`: `ColumnTransformer` [TF-IDF (1–2-grams, 5,000 features) ± `StandardScaler`-scaled linguistic features] → classifier. Because all learned steps are inside the pipeline, they are fitted only on training folds (**no data leakage**).
- A safeguard that blocks scaled (possibly negative) features from being fed to Multinomial Naive Bayes.

### `notebooks/NLP_Project1_AI_vs_Human_Clean.ipynb`
- Rebuilt the earlier notebook. The old version fitted TF-IDF on the whole dataset **before** the train/test split. In the new version the holdout is split off first, and every transformation is fitted inside the pipelines.
- Colab-friendly file upload and support for both semicolon- and comma-separated CSV files.

## 3. Model training
- Implemented the training loop in `src/train.py`: running `cross_validate` for each experiment, collecting accuracy and macro-F1 (mean and SD), sorting by macro-F1 and refitting the best pipeline on the full training split.
- Ran the training on the main corpus (3,382 texts) and the diagnostic set (200 texts) and checked that the script results match the notebook.

## 4. Documentation
- **`docs/preprocessing.md`:** the data collection, the input format, every cleaning step and the reason for it, the leakage-prevention design and the definitions of the linguistic features.
- Module docstrings and usage examples in `src/preprocess.py`, `src/features.py` and `src/train.py`.
- `requirements.txt` and `.gitignore`.

## Files I mainly worked on
| File | Work |
| --- | --- |
| `src/features.py` | Linguistic feature transformer and pipeline factory |
| `src/train.py` | Training loop and final refit |
| `notebooks/NLP_Project1_AI_vs_Human_Clean.ipynb` | Leakage-free Colab notebook |
| `docs/preprocessing.md`, `requirements.txt` | Technical documentation |

## Challenges and solutions
- **Data leakage:** the first notebook fitted TF-IDF before splitting. Moving every learned transformation into a scikit-learn `Pipeline` fixed this.
- **Sparse and dense features together:** a `ColumnTransformer` applies TF-IDF and the linguistic features to the same `Text` column and scales only the dense features.
