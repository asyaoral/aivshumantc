# Dataset provenance

The final experiment uses four separate datasets:

- The training dataset contains the original project corpus, 1,000 matched HC3 pairs, and 3,009 Heterogeneous AI Spans pairs.
- The external test contains 500 AI and 500 human texts and is kept separate from training.
- The HIP books challenge contains 400 AI and 400 human literary examples.
- The Heterogeneous spans challenge contains 500 mixed-authorship and 500 matched human-control texts. Its 123 source groups are completely excluded from training.

All labels follow `0 = AI or mixed authorship` and `1 = Human`. Exact normalized text overlap across all four datasets is checked by the code.

Before redistributing or reusing the texts, verify the license and attribution requirements of every original source. The software license does not automatically grant rights to the datasets.
