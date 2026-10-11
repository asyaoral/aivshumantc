# Dataset licensing and third-party notices

The MIT license in `LICENSE` applies only to software and documentation written by the project team. It does **not** place all text contained in the CSV files under the MIT license.

The raw and processed CSV files are research compilations assembled from sources with different terms. Redistribution of this repository does not replace or override those terms. Anyone reusing the data must review the original source terms and preserve required attribution.

## Source-specific terms

| Component | Source terms and reuse notice |
| --- | --- |
| Original team corpus | Contains team-created labels and AI-generated counterparts together with human passages collected from course books, articles, comments, and similar sources. The project does not claim ownership of third-party human wording. Reusers must obtain permission or rely on an applicable legal exception for those passages. |
| Kaggle LLM Detect AI Generated Text | Distributed through the [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text). No separate permissive license is asserted here. Users should obtain the source data from Kaggle and comply with the competition rules. |
| HC3 | The [HC3 dataset card](https://huggingface.co/datasets/Hello-SimpleAI/HC3) identifies the dataset license as **CC BY-SA 4.0**. Attribution and share-alike requirements apply. |
| HIP training and evaluation data | The [HIP dataset card](https://huggingface.co/datasets/YixuanEvenXu/HIP-training-and-evaluation-data) uses `license: other` because it combines RAID, MAGE, and OpenAI-generated paraphrases. Users must comply with the upstream dataset and provider terms. |
| Heterogeneous AI Spans | The [dataset card](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans) uses `mixed-source-terms`. Its [rights notice](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans/blob/main/LICENSE.md) states that no single permissive content license covers all source wording. Per-source attribution and rights information remain applicable. |

## Derived files

The cleaned training CSV and all evaluation CSV files remain subject to the same upstream terms as the raw text from which they were derived. Cleaning, relabelling, filtering, pairing, or format conversion does not create new ownership of the underlying text.

The project team's original code is available under the MIT license. Team-created selection metadata, labels, and documentation may be reused only to the extent that the contributors hold the necessary rights. This notice is not legal advice.
