# Academic Presentation Script & Technical Defense Guide

**Course:** COE025 — Natural Language Processing | Project 1  
**Project:** AI vs. Human Text Classification  
**Target Delivery Time:** **1 minute 45 seconds – 1 minute 55 seconds** (Under 2-minute professor requirement)  
**Slide Deck:** `presentation/final_presentation.pptx` (4 slides, 16:9 widescreen)

---

## 1. Timing Breakdown & Rehearsal Schedule

| Slide | Section | Target Time | Cumulative | Spoken Word Count |
|:---:|:---|:---:|:---:|:---:|
| **Slide 1** | Problem Statement & Preprocessed Dataset | 0:00 – 0:25 (25s) | 0:25 | 52 words |
| **Slide 2** | Methodology, Leakage Prevention & Metric Choice | 0:25 – 0:50 (25s) | 0:50 | 56 words |
| **Slide 3** | Benchmark Results, Ablation & Holdout Test | 0:50 – 1:18 (28s) | 1:18 | 61 words |
| **Slide 4** | Literature Comparison, Limitations & Next Steps | 1:18 – 1:45 (27s) | **1:45** | 53 words |
| **Total** | **Full Academic Presentation** | **1 min 40s – 1 min 48s** | — | **222 words** |

> **Delivery Guidance:**  
> - Total spoken word count: **222 words** (strictly within the 220–230 word requirement).  
> - Speaking rate: ~130–135 words per minute, yielding an estimated delivery time of **1 minute 42 seconds to 1 minute 47 seconds** (with 1-second slide pauses).  
> - Designed in fluent, natural academic English for effortless oral delivery without verbatim reading.

---

## 2. Slide-by-Slide Presentation Script

### Slide 1: Problem & Data (0:00 – 0:25 | 52 words)
*Visual Focus: 3,382 clean passages (1,698 AI / 1,684 Human), 50-word threshold.*

> **"Good morning. This project investigates binary text classification to distinguish AI-generated from human-written English text. Starting from our augmented dataset of 3,402 passages, our reproducible preprocessing pipeline filters short texts under 50 words and eliminates exact label conflicts and duplicates. This yields 3,382 clean, balanced samples: 1,698 AI-generated and 1,684 human passages."**

- **Transition Cue:** *"To ensure scientific rigor, we designed an evaluation protocol that strictly prevents data leakage."*

---

### Slide 2: Methodology & Leakage Prevention (0:25 – 0:50 | 56 words)
*Visual Focus: 80/20 stratified split, ColumnTransformer, fold-local TF-IDF, 5-fold CV.*

> **"To prevent data leakage, we reserve an untouched 20% holdout and evaluate candidates using stratified 5-fold cross-validation on the 80% training set. We extract TF-IDF unigrams and bigrams up to 5,000 features, fitted strictly inside each training fold. We explicitly chose Macro-F1 because falsely accusing human authors and missing AI-generated texts carry equal, severe real-world consequences."**

- **Transition Cue:** *"Comparing our candidate models under identical cross-validation conditions yields clear results."*

---

### Slide 3: Benchmark & Ablations (0:50 – 1:18 | 61 words)
*Visual Focus: Benchmark table (Linear SVM 0.9896, LogReg 0.9834, NB 0.8904), Holdout metrics (98.23%).*

> **"Comparing our baselines across training folds, Linear SVM achieved the highest Macro-F1 of 0.9896, followed closely by Logistic Regression at 0.9834, while Naive Bayes trailed at 0.8904. Ablation experiments showed that removing stop words or adding six handcrafted linguistic features did not improve SVM performance. On the untouched 677-sample holdout, our selected Linear SVM achieved 98.23% accuracy with exactly 12 misclassifications."**

- **Transition Cue:** *"While these metrics are high, they must be interpreted within a rigorous scientific context."*

---

### Slide 4: Scientific Interpretation & Limitations (1:18 – 1:45 | 53 words)
*Visual Focus: Literature connections (HC3, SemEval-2024, RAID), out-of-domain caution.*

> **"However, high in-corpus accuracy does not make a universal detector. Published academic benchmarks such as HC3, SemEval-2024, and RAID demonstrate that n-gram detectors frequently exploit corpus-specific topics and degrade under distribution shift or unseen generators. Rigorous future work requires cross-domain testing and source-aware splitting. Thank you, and I look forward to your questions."**

---

## 3. Anticipated Technical Defense Q&A

### Q1: Why did you choose Macro-F1 over Accuracy as the primary selection metric?
**Defense:**
- Macro-F1 computes the harmonic mean of precision and recall independently for each class and calculates their unweighted average:
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{AI}} + \text{F1}_{\text{Human}}}{2}$$
- In real-world AI detection, asymmetric errors carry severe and distinct consequences:
  1. **False Positives (Human flagged as AI):** Falsely accusing a student or author of academic dishonesty causes immediate reputational and disciplinary harm.
  2. **False Negatives (AI missed as Human):** Undermines the integrity of verification systems.
- While Accuracy can mask severe class degradation when class distributions shift, Macro-F1 guarantees that both classes contribute equally to model selection.

---

### Q2: How did you mathematically and programmatically prevent data leakage?
**Defense:**
- **The Pitfall:** Standard leakage occurs when text vectorization (computing document frequencies, inverse document frequency weights, and vocabulary dictionaries) or feature scaling is performed on the whole dataset prior to splitting or cross-validation.
- **Our Prevention Architecture:**
  1. Data is split once into 80% training ($N=2{,}705$) and 20% holdout ($N=677$) using `train_test_split(..., stratify=y, random_state=42)`.
  2. The holdout is locked and never seen during model selection.
  3. Preprocessing, TF-IDF vectorization, and standard scaling are wrapped in a single `sklearn.pipeline.Pipeline` using `ColumnTransformer`.
  4. During `cross_validate(cv=StratifiedKFold(n_splits=5))`, `pipeline.fit()` is executed strictly on the $K-1$ training folds (approx. 2,164 samples) in each iteration. The validation fold and the holdout set remain completely out-of-sample.

---

### Q3: Why did appending the six linguistic features not improve the Linear SVM?
**Defense:**
- The six handcrafted features include: Average Sentence Length, Lexical Diversity (Type-Token Ratio), Punctuation Density, Capitalization Ratio, Pronoun Ratio, and Sentence Count.
- In our controlled ablation, adding these features slightly decreased mean CV Macro-F1 from **0.9896** to **0.9885** (and increased variance from $\pm 0.0022$ to $\pm 0.0049$).
- **Reason:** A 5,000-dimensional word unigram and bigram TF-IDF space already implicitly captures punctuation tokens, lexical variety, and function words with higher spatial granularity. Adding 6 global scalar features—even after standard scaling—introduces redundant, low-dimensional noise that slightly shifts the optimal maximum-margin hyperplane in high-dimensional linear SVM space.

---

### Q4: Why did removing English stop words slightly decrease CV performance (0.9896 → 0.9893)?
**Defense:**
- In topical search or document indexing, stop words are often discarded. However, in **stylometry and authorship attribution**, stop words (articles, prepositions, auxiliary verbs, conjunctions) are critical markers of syntactic structure and pacing.
- Large language models generate text using token probability distributions that exhibit subtle, consistent differences in function-word transitions compared to human writers. Preserving stop words retains these distinctive stylometric signals.

---

### Q5: Can we compare your 0.9823 holdout score with published benchmarks like HC3 or RAID?
**Defense:**
- **No direct numerical ranking is scientifically valid.**
- Published academic benchmarks evaluate under different conditions:
  - **HC3 (Guo et al., 2023):** Evaluated domain-specific Q&A pairs (medical, legal, computer science) comparing ChatGPT against human experts.
  - **SemEval-2024 Task 8 (Pulipaka et al., 2024):** Demonstrated that traditional and neural models trained on specific writing prompts suffer substantial accuracy drops (often 20–30%) when tested on unseen generator architectures or novel domains.
  - **RAID (Dugan et al., 2024):** Benchmarked detectors across 8 LLM families and multiple adversarial attacks, proving that superficial n-gram patterns fail under adversarial rewriting.
- Our 98.23% result demonstrates strong **within-corpus** discriminative power on this specific augmented dataset; claiming it as a general detector would be misleading.

---

### Q6: Why was the 200 flipped-label dataset kept strictly separate?
**Defense:**
- The file `train_essays_200_flipped_labels.csv` contains intentionally inverted or perturbed class labels designed for diagnostic error analysis.
- Merging noisy or corrupted labels into the primary training set would violate empirical risk minimization, injecting systematic label noise and corrupting the benchmark. Keeping it quarantined preserves reproducible integrity.

---

### Q7: What characterized the 12 misclassifications on the holdout test set?
**Defense:**
- On the 677-sample holdout, the Linear SVM achieved 665 correct predictions and 12 errors (error rate: 1.77%):
  - **6 False Positives (Human text predicted as AI):** Passages written with highly uniform sentence length, formal academic vocabulary, and rigid grammatical syntax that mirrored LLM output patterns.
  - **6 False Negatives (AI text predicted as Human):** AI passages prompted with conversational framing, varied sentence structures, or colloquialisms that evaded the linear boundary.
