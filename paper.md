---
title: 'leakcheck: screening reported classification results for the resampling-before-splitting artefact'
tags:
  - Python
  - data leakage
  - class imbalance
  - SMOTE
  - peer review
  - reproducibility
  - research software
authors:
  # JOSS review is public: replace these placeholders before submission.
  - name: Author names withheld for review
    corresponding: true
    affiliation: 1
affiliations:
  - index: 1
    name: Affiliations withheld for review
date: 8 September 2026
bibliography: paper.bib
---

# Summary

Applying SMOTE [@chawla2002] or a related over-sampler to a whole dataset and only then splitting it into training and test partitions is a documented form of data leakage in applied machine learning [@kaufman2012; @kapoor2023; @vandewiele2021; @apicella2025]. Because synthetic observations may then be constructed using information from cases assigned to the test partition, the resulting score can overestimate performance on genuinely unseen cases.

`leakcheck` is a small Python package that screens a reported classification result for one identifiable signature of that problem. It is designed for readers rather than authors: it requires no access to code or data, but relies on quantities that papers commonly report. The user supplies the headline minority-class F1, the imbalance ratio, the number of minority cases, the number of features, and the model family. The package returns a model-estimated screening probability, a verdict band, and a short written explanation suitable for a review.

The screening model is a gradient-boosted classifier trained on 22,400 labelled results from 41 public datasets in which the same models were evaluated under both orderings, so the experimental condition is known by construction. Leave-one-dataset-out validation gives an area under the ROC curve (AUC) of 0.795 and a Brier score of 0.197 [@brier1950]. The output is a triage signal rather than a causal finding of leakage.

```python
import leakcheck
r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                    n_features=44, clf='RandomForest')
print(r.probability, r.verdict)
print(r.explain())
```

# Statement of need

That error sits inside a broader reproducibility problem: leakage, non-replicable benchmarks, and overstated conclusions remain common in applied machine learning [@kapoor2024; @semmelrock2025; @lones2024; @pineau2021; @varoquaux2022]. Reviewers and editors may lack the time, access, or computational environment needed to run an author's code, and existing guidance is often too qualitative to resolve questions about a specific reported result.

`leakcheck` turns the general warning into a question about the case in hand: given what this paper reports, how unusual is this number relative to a controlled corpus in which resampling order is known? The package is intended for three audiences:

- **Peer review.** Decide whether to ask the authors where the resampler sits in their pipeline. The tool prints a paragraph suitable for pasting into a report, which states that the output is a screen rather than a finding.
- **Self-audit.** Check one's own results before submission.
- **Meta-research.** Screen a corpus of published results; a single command accepts a table.

The empirical basis is a regularity in the companion study: train-only scores varied more strongly with problem difficulty, whereas resample-then-split scores concentrated near the upper end of the observed range (variance ratio 4.86). That pattern is consistent with the class-imbalance literature [@he2009; @fernandez2018; @goorbergh2022; @elor2022; @elreedy2024; @carriero2025].

# State of the field

Existing tooling for leakage generally operates on code or data. Validation libraries can detect some forms of train/test contamination when they can inspect both partitions. Reproducibility checklists [@kapoor2023; @kapoor2024] support better practice but do not by themselves evaluate a number already in print. Splitters such as DataSAIL [@joeres2025] can reduce specific leakage risks when the analyst has access to the data. Pipeline objects in imbalanced-learn [@lemaitre2017] facilitate the correct prospective ordering of resampling and splitting.

Those tools are the right contribution when code or data are available. `leakcheck` addresses a different use case: retrospective screening from published summary information. Building a new package rather than extending imbalanced-learn or a validation library is therefore a deliberate choice. Those ecosystems assume access to the pipeline; a reviewer of a PDF typically has only the reported F1, a few dataset descriptors, and a model name. The scholarly contribution is a fitted, inspectable screen for that setting, with the limitations that follow from not observing the pipeline.

# Software design

Each matched pair in the companion corpus yields two labelled examples—a resample-then-split result and a train-only result—matched on the quantities a paper would report except for the resulting score. A gradient-boosted classifier is trained to separate them from the reported F1, the log imbalance ratio, the log minority count, the log feature count, and the model family. Validation is leave-one-dataset-out, so each held-out prediction comes from a model fitted without that dataset.

The corpus contains 11,200 matched comparisons (22,400 labelled records). Within a pair, only the position of resampling relative to the train/test split changes. The 41 source datasets span 208 to 5,000 rows, 3 to 180 features, 30 to 1,813 minority cases, and imbalance ratios from 1.02 to 109.94. Six concern employee attrition; the remaining 35 are from the Penn Machine Learning Benchmarks [@olson2017]. Classifier families are 5-nearest neighbours, decision trees, gradient boosting, logistic regression, random forests, RBF support-vector machines, and XGBoost. Resamplers are ADASYN, Borderline-SMOTE, random over-sampling, and SMOTE.

Four deployment choices follow from the intended user being a reviewer, not a modeller.

**The model ships as JSON, not as a pickle.** A pickled scikit-learn estimator couples deployment to particular library versions. The screen is instead exported as JSON—thresholds, split features, and leaf values—and evaluated by a pure-NumPy tree walker. This keeps run-time dependencies small and the stored model inspectable. Fixtures check that the walker reproduces the original scikit-learn predictions to within $10^{-9}$; `leakcheck.selftest()` runs that check on an installed copy.

**Unknown model families resolve rather than fail.** Papers name their models in many ways. The package maps around thirty spellings onto the seven families in the training corpus, and an unrecognised name falls back to the tree-ensemble family with the fallback documented, rather than raising.

**Extrapolation is announced.** Inputs outside the fitted ranges—imbalance ratios beyond 1.02 to 109.9, or minority counts outside 30 to 1,813—still return a model output, but the result carries an explicit note that the answer is an extrapolation.

**The output is a question, not a verdict.** No verdict string asserts that an error occurred; the strongest label is “likely leaked”, and the explanation recommends asking where the resampler sits. Bands below 0.35, from 0.35 to 0.65, and above 0.65 are operational triage categories, not diagnostic cut-offs. An ablation that removes the reported F1 and leaves only the dataset descriptors falls to an AUC of 0.500: the signal depends on the relationship between the score and the descriptors, not on the descriptors alone.

The signature is not causally unique. Target leakage, related records crossing the split, test-set tuning, an unusually separable dataset, or a macro-averaged F1 reported as a minority-class F1 can produce the same pattern. Conversely, resampling-before-splitting need not always yield a conspicuously high score. The screen is a triage instrument with false positives and false negatives, not a deterministic leakage detector.

# Research impact statement

The software is used in the companion methodological study that produced its training corpus, and it is the artefact through which that study's screening result is distributed. Leave-one-dataset-out evaluation yields an AUC of 0.795, accuracy 0.697 at the natural threshold, and a Brier score of 0.197. Per-dataset AUC has median 0.987 and minimum 0.388. Calibration is closer in the tails than in the middle of the range, which is why the package reports verdict bands rather than treating the number as a precisely calibrated probability for an individual paper.

A headline F1 of 0.92 on the IBM HR benchmark—237 leavers among 1,470 employees, imbalance 5.2 to 1, 44 encoded features, a random forest—falls in the “likely leaked” band. Near-perfect scores on small imbalanced HR tables merit scrutiny [@tambe2019], but the appropriate response is to ask where the resampler sits relative to the split, not to infer that a paper is wrong.

Applied to headline minority-class F1 values from recent employee-attrition studies [@hamja2025; @nassreddine2026; @baydili2025; @alali2026; @alyousef2026; @cavescu2026], the two studies that describe train-only resampling [@alyousef2026; @cavescu2026] fall in the unremarkable band, while a study that states that resampling preceded the split [@baydili2025], and several that do not state the ordering, receive higher scores. This is an illustration of the intended review use, not an estimate of prevalence. Protocol labels follow what each paper states; “not stated” is not a ground-truth leakage label.

The package is installable from source, carries a machine-readable citation file, and is released under the BSD 3-Clause licence. Tests cover input validation, alias resolution, score-sensitivity checks, batched versus single-record agreement, the command-line interface, and stored regression fixtures. Source, tests, documentation, and the exported model are archived on Zenodo at [10.5281/zenodo.22196497](https://doi.org/10.5281/zenodo.22196497).

# AI usage disclosure

Generative AI (Cursor Grok 4.6) was used to condense the authors' source manuscript into this JOSS Markdown file, including mapping the longer paper onto the required JOSS sections and assembling the companion BibTeX file. All scientific claims, reported numbers, and design decisions were taken from the source manuscript. The authors reviewed and edited the adapted text and take responsibility for its accuracy, originality, and completeness. Generative AI was not used to produce the experimental corpus, fit the screening model, or invent results.

# Acknowledgements

No external funding is declared. The training corpus uses publicly available datasets from the Penn Machine Learning Benchmarks and public employee-attrition tables.

# References
