---
title: 'leakcheck: screening reported classification results for the resampling-before-splitting artefact'
tags:
  - Python
  - data leakage
  - class imbalance
  - SMOTE
  - reproducibility
  - peer review
  - research software
authors:
  - name: Hieu Le Ngoc
    orcid: 0000-0002-1133-1433
    email: lnhieu@ptit.edu.vn
    affiliation: 1
    corresponding: true
  - name: Trung Huynh Ngoc Thanh
    orcid: 0009-0001-5785-6630
    email: hnttrung@ufm.edu.vn
    affiliation: 2
affiliations:
  - name: Faculty of Information Technology 2, Posts and Telecommunications Institute of Technology
    index: 1
  - name: Faculty of Data Science, University of Finance – Marketing
    index: 2
date: 2026
bibliography: paper.bib
---

# Summary

`leakcheck` estimates whether a published classification result resembles the
statistical pattern produced, in a controlled corpus, by applying a synthetic
over-sampler to a dataset *before* splitting it into training and test
partitions rather than to the training partition alone. It takes only
quantities that papers commonly report -- the headline minority-class F1, the
imbalance ratio, the number of minority cases, the number of features and the
model family -- and returns a bounded screening score together
with a short written explanation.

The screening model is a gradient-boosted classifier trained on 22,400 labelled
results from 41 public datasets in which the same models were evaluated under
both orderings, so the experimental condition is known by construction;
leave-one-dataset-out validation gives an AUC of 0.795 and a Brier score of
0.197. The output is a triage signal rather than a causal finding of leakage.
The trained model ships as plain JSON and is evaluated by a pure-NumPy tree
walker, limiting run-time dependencies and reducing sensitivity to changes in
modelling-library serialization.

# Statement of need

Applying SMOTE [@chawla2002] or a related over-sampler to a whole dataset and
only then splitting it is a documented form of data leakage in applied machine
learning [@kaufman2012; @kapoor2023; @vandewiele2021]. Because synthetic
observations may then be constructed using information from cases assigned to
the test partition, the resulting score can overestimate performance on
genuinely unseen cases.

Reviewers and editors may lack the time, access, or computational environment
needed to run an author's code. Existing guidance is often qualitative.
`leakcheck` turns the general warning into a question about the case in hand:
given what this paper reports, how unusual is this number relative to the
controlled corpus?

The empirical basis is a regularity observed in the companion study. Within
that corpus, train-only scores varied more strongly with problem difficulty,
whereas resample-then-split scores were more concentrated near the upper end of
the observed range. The variance of the train-only scores was 4.86 times that
of the leaked scores.

`leakcheck` is intended for three uses:

1. **Peer review.** Decide whether to ask the authors where the resampler sits
   in their pipeline. The tool prints a paragraph suitable for pasting into a
   report, which states that the output is a screen rather than a finding.
2. **Self-audit.** Check one's own results before submission.
3. **Meta-research.** Screen a table of extracted published results to
   prioritise cases for methodological review; screening flags alone do not
   estimate leakage prevalence.

# A worked example

```python
import leakcheck

r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                    n_features=44, clf="RandomForest")
print(r.score, r.verdict)
print(r.explain())
```

From the command line:

```console
$ leakcheck --f1 0.92 --ir 5.2 --n-minority 237 --n-features 44 \
            --clf RandomForest
```

# What the tool deliberately does not do

`leakcheck` is trained to recognise the empirical signature of resampling
before splitting; it does not observe the pipeline that produced a reported
score. Several non-leakage explanations can produce the same signature,
including a macro-averaged F1 reported as a minority-class F1, unusual
preprocessing, or a genuinely separable dataset. No verdict string asserts that
an error occurred; the strongest package label is "high screening score - methods check recommended", and the
accompanying explanation recommends asking where the resampler sits. Inputs
outside the fitted ranges still return a screening score, but the result carries an
explicit note that the answer is an extrapolation.

# State of the field

Existing tooling for leakage generally operates on code or data. Validation
libraries can detect some forms of train/test contamination when they can
inspect both partitions. Reproducibility checklists [@kapoor2023] support
better practice but do not by themselves evaluate a number already in print.
Pipeline objects in imbalanced-learn [@lemaitre2017] facilitate the correct
prospective ordering of resampling and splitting. `leakcheck` addresses a
different use case: retrospective screening from published summary information.

# Quality control

The package has no dependency beyond NumPy at run time. The test suite covers
input validation, model-family alias resolution, selected checks of screening-score
behaviour as the reported score changes, agreement between batched and
single-record evaluation, the command line interface, and stored regression
fixtures verified to reproduce the original scikit-learn predictions to within
$10^{-6}$. `leakcheck.selftest()` runs the fixture check on an installed copy.
The package is installable from source with a standard build backend and
carries a machine-readable citation file.

# Acknowledgements

This research is funded by University of Finance – Marketing, HCMC Vietnam.
This study is supported by Posts and Telecommunications Institute of Technology
(PTIT), Vietnam. The screening model is trained on the 41-dataset corpus
shipped in `data/processed/` of this repository.

# Revision status

Version 1.0.1 is a local revision. Dataset-cluster intervals and matched baselines are provided in `results/paper2b_revision_v1`. External diagnostic validation and a verified new archive are outstanding. The historical archive identifier below does not establish deposit of this revision.

# Data availability

The source code and associated materials are available from the
[occbuu/leakcheck](https://github.com/occbuu/leakcheck) repository and the original manuscript cites the archive identifier
[10.5281/zenodo.22196497](https://doi.org/10.5281/zenodo.22196497).

# References
