---
title: "Leakage makes every problem look equally easy: A multi-dataset measurement of the resampling-before-splitting artefact"
journal: Expert Systems with Applications
article-type: Research Paper
corresponding: Hieu Le Ngoc
bibliography: paper.bib
---

<!--
ESWA uses double-anonymized peer review.
Upload (1) this title page with authors and (2) a blinded manuscript that
omits names, acknowledgements, funding, and repository URLs.
Highlights are in highlights.md (mandatory Elsevier file, ≤85 characters each).
-->

# Leakage makes every problem look equally easy: A multi-dataset measurement of the resampling-before-splitting artefact

Hieu Le Ngoc^{1,*}, Thuy Truong Dinh Hai^{2}

^{1} Faculty of Information Technology 2, Posts and Telecommunications Institute of Technology, Vietnam  
ORCID: [0000-0002-1133-1433](https://orcid.org/0000-0002-1133-1433)  
E-mail: <lnhieu@ptit.edu.vn>

^{2} Faculty of Data Science, University of Finance – Marketing, Ho Chi Minh City, Vietnam  
ORCID: [0009-0006-2946-7490](https://orcid.org/0009-0006-2946-7490)  
E-mail: <tdh.thuy@ufm.edu.vn>

\* Corresponding author.

## Highlights

- Resampling before splitting inflates minority-class F1 by 0.224 on 41 datasets.
- Leaked F1 has 4.86x less variance and rises with imbalance, unlike honest F1.
- The leakage premium is almost fully predicted by problem difficulty (R2=0.795).
- A screen using reported F1 and descriptors ranks leaked results at AUC 0.795.
- Targeting 25% minority prevalence instead of full balance cuts the artefact 25%.

## Abstract

Intelligent HR and credit-risk systems routinely report near-perfect minority-class F1 after synthetic over-sampling. A common implementation error—applying SMOTE or a related over-sampler *before* the train–test split rather than to the training partition alone—lets interpolated copies of held-out cases enter training, so the reported score no longer describes performance on unseen employees or applicants. We measure that end-to-end workflow artefact on 41 public binary-classification datasets (imbalance ratios 1.02–109.9), including six employee-attrition tables, with seven classifier families, four over-samplers and ten seeds (11,200 matched leaked-versus-honest comparisons). Mean inflation of minority-class F1 is 0.224. Leakage does not merely raise scores: it destroys the relationship between score and problem difficulty. Under the correct ordering, F1 falls as imbalance rises; under the leaked ordering it collapses onto a narrow band (4.86 times less variance) and *rises* with imbalance ($r=0.578$) where the honest score falls ($r=-0.268$). The premium is therefore predicted by how hard the problem actually is ($R^2=0.795$). A contamination rate traces the mechanism; resampling to a 0.25 minority prevalence rather than to full balance cuts the artefact by 25%. Because the signature is stable, a published result can be screened from quantities papers already report (leave-one-dataset-out AUC 0.795). We release the corpus, experimental record and screening tool so that editors, reviewers and HR-analytics practitioners can treat a high F1 on a hard imbalanced table as a methods question rather than a finding.

**Keywords:** data leakage; class imbalance; SMOTE; employee attrition; HR analytics; reproducibility; expert systems evaluation

## 1. Introduction

A machine learning result is a claim about a procedure applied to data that the procedure has not seen. Data leakage is the failure of that condition, and it is now recognised as a leading cause of results that do not replicate [@kaufman2012; @kapoor2023; @whalen2022; @roth2026grammar; @roth2026]. Among its many forms, one is unusually easy to commit and unusually easy to overlook: applying a synthetic over-sampler such as SMOTE [@chawla2002] to a dataset and only then splitting it into training and test partitions. Two lines of code in the wrong order produce a pipeline that looks correct, runs without warning, and reports a number that describes something other than what the author believes it describes.

The mechanism is not subtle once stated. SMOTE creates a new minority point by interpolating between a real minority point and one of its neighbours. If the interpolation happens before the split, some of those synthetic points inherit most of their coordinates from a real point that the split then assigns to the test set, while the synthetic point itself lands in the training set. The model does not have to generalise to those test cases. It has to remember them.

That the error inflates results has been stated many times and demonstrated at length on a medical corpus [@vandewiele2021]. A recent quantitative landscape has also measured oversampling leakage within a much broader taxonomy of leakage mechanisms [@roth2026]. What remains unresolved is whether this particular workflow error *replaces* the expected relationship between score and problem difficulty, how its end-to-end premium relates to contamination and test-distribution change, and whether its signature is stable enough to be recognised in results one did not produce.

Those questions matter for expert systems that score people. Predictive HR analytics, credit scoring and related decision-support tools increasingly rest on imbalanced tabular classifiers [@tambe2019]. A reviewer confronted with a reported F1 of 0.95 on a problem with two hundred minority leavers has no way to act on a qualitative warning. An editor deciding whether to request code has nothing to weigh. A field trying to understand why its published numbers cluster in a suspiciously narrow range needs a quantitative account of why that would happen.

This paper supplies the measurement. We assembled 41 public binary classification problems, 35 drawn from a standard benchmark collection [@olson2017; @romano2022] and 6 from the employee-attrition literature, with imbalance ratios from 1.02 to 109.9, sample sizes from 208 to 5,000 and between 3 and 180 predictors. On each we fitted the same model three ways, changing nothing but where the resampler sits: before the split, after the split on the training partition only, and not at all. Seven classifier families, four over-samplers and ten seeds give 11,200 matched leaked-versus-honest comparisons from 25,340 fitted models.

The headline number is the smaller half of the finding. The larger half is a regularity we did not expect and have not found reported: leakage does not shift the score distribution upward so much as it flattens it. Under the correct ordering, F1 behaves as a measurement should, falling as the problem gets harder. Under the leaked ordering it stops responding to difficulty almost entirely, landing in a narrow band whose position depends mainly on which classifier was used. An easy problem and a hard problem, run through the same faulty pipeline, come out looking alike.

This inverts the usual intuition. The common description of leakage is that it makes results “a bit too good”. That description is wrong in a way that matters: leakage does not add a bias to a measurement, it replaces the measurement with a different one. The reported quantity stops being a property of the problem and becomes a property of the model class.

Three contributions follow.

- **C1 – Focused measurement.** We characterise resampling-before-splitting as an end-to-end workflow artefact across a corpus spanning two orders of magnitude in imbalance, seven algorithm families and four over-sampling methods, emphasising its variance collapse and reversal of the score–difficulty relationship rather than claiming the first measurement of leakage magnitude.
- **C2 – A regularity and its mechanism.** We show that the leaked score is close to invariant to problem difficulty and, where it does respond, responds in the wrong direction; that the premium is therefore predictable from difficulty alone; and that both follow from a measurable contamination rate. A controlled simulation separates imbalance, dimensionality, class overlap and sample size, which real data confounds.
- **C3 – A screening tool.** Because the artefact has a stable signature, a published result can be screened from what papers already report. We train such a screen, validate it leave-one-dataset-out, and release it with the experimental archive.

A companion study by the present authors measured this effect on a single attrition benchmark and concluded that replication on independent data was needed before the finding could be generalised. The present work is that replication, at a scale that turns a case report into a measurement, and it asks the further questions the single-dataset design could not: whether the effect generalises, what governs its size, and whether it can be recognised from the outside.

## 2. Related work

### 2.1 Leakage

Kaufman and colleagues [@kaufman2012] gave the first systematic treatment, defining leakage as the introduction of information about the target that would not be legitimately available at prediction time, and cataloguing the forms it takes. Two decades later, Kapoor and Narayanan [@kapoor2023] collated twenty-two review papers covering seventeen scientific fields and identified leakage affecting at least 294 published studies. Whalen and colleagues [@whalen2022] make the same case for genomics. Recent work has extended the evidence to workflow enforcement, public leaderboards, and related evaluation failures [@roth2026grammar; @roth2026].

Most directly, Roth [@roth2026] reports 28 within-subject leakage experiments across 2,047 tabular datasets and includes oversampling-before-splitting among the memorisation mechanisms studied. That work establishes a broad quantitative landscape, including capacity and sample-size effects. Our narrower contribution is to characterise the resulting F1 distribution: its loss of sensitivity to difficulty, its reversal with imbalance, its relationship to a measurable contamination rate, and its detectability from quantities available to a reader. Vandewiele and colleagues [@vandewiele2021] provide the closest detailed domain case, reproducing inflated preterm-birth results and showing that they collapse under correct evaluation.

### 2.2 Synthetic over-sampling and its sceptics

SMOTE [@chawla2002] and its descendants Borderline-SMOTE [@han2005] and ADASYN [@he2008] are the most widely used response to class imbalance. Reviews document both the breadth of the resulting family and the persistent problems caused by overlap, noise and inappropriate synthetic samples [@he2009; @fernandez2018]. A steady line of work questions whether rebalancing helps. Blagus and Lusa [@blagus2013] showed that SMOTE is largely ineffective in high dimensions. Van den Goorbergh and colleagues [@goorbergh2022] demonstrated that imbalance corrections destroy the calibration of risk prediction models while leaving discrimination unchanged, a result Carriero and colleagues [@carriero2025] extended to machine-learning models by simulation. Large empirical comparisons likewise find that no rebalancing can be competitive when the classifier is appropriately selected or tuned [@elor2022]. Our results are consistent with this line: across the whole corpus, no resampling beat correctly-ordered resampling in 34.7% of matched comparisons.

### 2.3 Large-scale empirical benchmarking

The practice of evaluating a claim across many datasets rather than a favoured few goes back to Demšar [@demsar2006]. PMLB [@olson2017; @romano2022] curates a large collection of standardised classification problems for exactly this purpose. Our design borrows the logic: the question is not whether the artefact appears in a particular case but how its magnitude behaves as a function of dataset properties.

### 2.4 The gap

Three things are missing from the literature above. First, no measurement of the effect size of resampling-before-splitting as a function of dataset characteristics, with a matched end-to-end workflow estimand. Second, no account of what the leaked score is actually measuring when it is not measuring the problem. Third, no way for a reader of an expert-system paper to act on the concern in a specific reported result. This paper addresses all three.

## 3. Method

### 3.1 Corpus

We assembled 41 binary classification problems in two groups. Group A consists of 35 benchmark problems derived from PMLB [@olson2017; @romano2022]. Where a source is multi-class we binarise it one-versus-rest on the rarest class with at least thirty cases. Group B consists of 6 employee-attrition and turnover datasets. We chose attrition as a deliberately coherent applied case study rather than as an additional convenience sample: it is a canonical imbalanced decision problem, resampling is common in its recent predictive literature, multiple studies reuse the same public benchmarks, and both false-positive and false-negative errors carry direct resource implications [@tambe2019]. The HR datasets also anchor the mild-imbalance end of the corpus (ratios 1.02 to 7.42). One dataset [@alqahtani2025] is survey data collected for the purpose rather than a re-publication of an existing corpus.

Two filters apply. A dataset enters only if it has at least thirty minority cases, because below that a 70:30 split leaves too few test positives for F1 to be stable. Datasets larger than 5,000 rows are stratified-subsampled to 5,000. The cap standardises the maximum effective sample size, prevents a small number of large problems from dominating computational cost, and defines the scope of the estimand: the results describe capped tabular problems, not enterprise-scale populations. Because leakage magnitude can vary with sample size [@roth2026], extrapolation beyond 5,000 observations requires a dedicated within-dataset scaling analysis. Both thresholds are exposed as parameters in the released code. Table 1 lists the corpus.

**Table 1.** The corpus. IR is the ratio of majority to minority cases after any subsampling. Construction records how a binary target was obtained.

| Dataset | Domain | *n* | *p* | Minority | IR | Construction |
|---|---|---:|---:|---:|---:|---|
| turnover_russia | HR | 1,129 | 50 | 558 | 1.02 | file |
| saudi_attrition | HR | 1,191 | 100 | 515 | 1.31 | file |
| job_change_ds | HR | 5,000 | 62 | 1,247 | 3.01 | file |
| ibm_hr | HR | 1,470 | 44 | 237 | 5.20 | file |
| hr_case_study | HR | 4,410 | 33 | 711 | 5.20 | file |
| healthcare_attr | HR | 1,676 | 40 | 199 | 7.42 | file |
| bupa | benchmark | 345 | 5 | 169 | 1.04 | binary |
| sonar | benchmark | 208 | 60 | 97 | 1.14 | binary |
| credit_approval_australia | benchmark | 690 | 15 | 307 | 1.25 | binary |
| spambase | benchmark | 4,601 | 57 | 1,813 | 1.54 | binary |
| ionosphere | benchmark | 351 | 33 | 126 | 1.79 | binary |
| magic | benchmark | 5,000 | 10 | 1,758 | 1.84 | binary |
| tic_tac_toe | benchmark | 958 | 9 | 332 | 1.89 | binary |
| titanic | benchmark | 2,207 | 8 | 711 | 2.10 | binary |
| credit_approval_germany | benchmark | 1,000 | 20 | 300 | 2.33 | binary |
| phoneme | benchmark | 5,000 | 5 | 1,467 | 2.41 | binary |
| haberman | benchmark | 306 | 3 | 81 | 2.78 | binary |
| dna | benchmark | 3,186 | 180 | 765 | 3.16 | one-vs-rest, class 2/3 |
| adult | benchmark | 5,000 | 14 | 1,196 | 3.18 | binary |
| contraceptive_method | benchmark | 1,473 | 9 | 333 | 3.42 | one-vs-rest, class 1/3 |
| clean2 | benchmark | 5,000 | 168 | 771 | 5.49 | binary |
| segmentation | benchmark | 2,310 | 18 | 330 | 6.00 | one-vs-rest, class 3/7 |
| churn | benchmark | 5,000 | 20 | 707 | 6.07 | binary |
| dermatology | benchmark | 366 | 34 | 49 | 6.47 | one-vs-rest, class 4/6 |
| ecoli | benchmark | 327 | 6 | 35 | 8.34 | one-vs-rest, class 4/5 |
| mfeat_karhunen | benchmark | 2,000 | 64 | 200 | 9.00 | one-vs-rest, class 9/10 |
| optdigits | benchmark | 5,000 | 62 | 493 | 9.14 | one-vs-rest, class 8/10 |
| satimage | benchmark | 5,000 | 36 | 486 | 9.29 | one-vs-rest, class 4/6 |
| pendigits | benchmark | 5,000 | 16 | 480 | 9.42 | one-vs-rest, class 3/10 |
| vowel | benchmark | 990 | 13 | 90 | 10.00 | one-vs-rest, class 7/11 |
| analcatdata_authorship | benchmark | 841 | 70 | 55 | 14.29 | one-vs-rest, class 2/4 |
| coil2000 | benchmark | 5,000 | 85 | 298 | 15.78 | binary |
| hypothyroid | benchmark | 3,163 | 25 | 151 | 19.95 | binary |
| car_evaluation | benchmark | 1,728 | 6 | 65 | 25.58 | one-vs-rest, class 3/4 |
| letter | benchmark | 5,000 | 16 | 184 | 26.17 | one-vs-rest, class 26/26 |
| wine_quality_white | benchmark | 4,898 | 11 | 163 | 29.05 | one-vs-rest, class 4/7 |
| ann_thyroid | benchmark | 5,000 | 21 | 115 | 42.48 | one-vs-rest, class 1/3 |
| allhyper | benchmark | 3,771 | 27 | 79 | 46.73 | one-vs-rest, class 2/4 |
| yeast | benchmark | 1,479 | 8 | 30 | 48.30 | one-vs-rest, class 7/9 |
| dis | benchmark | 3,772 | 27 | 58 | 64.03 | binary |
| allrep | benchmark | 3,772 | 27 | 34 | 109.94 | one-vs-rest, class 1/4 |

**Figure 1.** Coverage of the corpus. Each point is a dataset. The horizontal axis is the imbalance ratio and the vertical axis the number of minority cases, both on log scales. The employee-attrition datasets occupy the mild-imbalance corner; the benchmark problems extend the range by two orders of magnitude.

### 3.2 The three conditions

Every comparison in this paper is between pipelines that differ in exactly one respect. The data, the preprocessing, the classifier, its hyperparameters, the resampler, its random seed, the split proportion and the split seed are all held fixed. The only thing that moves is where the resampler sits.

**Table 2.** The three conditions. Everything except the position of the resampler is held identical.

| Condition | Procedure | Role |
|---|---|---|
| leaked | Resample the entire dataset, then split 70:30 | Reproduces the error as it appears in the literature |
| correct | Split 70:30, then resample the training partition only | The correct protocol |
| none | Split 70:30, no resampling | Baseline: is resampling helping at all? |

In the leaked condition the test partition is drawn from the resampled pool, so it contains synthetic points and has a different class prevalence and size from the real test partition. This is deliberate: it is what the erroneous pipeline actually does. The primary contrast is therefore an end-to-end workflow estimand. It combines training contamination, evaluation on synthetic observations and the prevalence shift induced by resampling; it should not be interpreted as the isolated causal effect of contaminated training data on a common test set. All features are standardised inside the pipeline, fitted on training data only in every condition.

### 3.3 A measure of the mechanism

We define the **contamination rate** as the fraction of held-out minority points whose nearest synthetic training point lies closer than half the median nearest-neighbour distance among real minority points. The reference distance is computed on the same standardised scale, so the measure is dimensionless and comparable across datasets. It is the mediating variable that turns an observation into an explanation, and it is what lets us say why the mitigation in Section 4.6 works.

### 3.4 The factorial

**Table 3.** Experimental design.

| Factor | Levels | Detail |
|---|---|---|
| Datasets | 41 | 35 benchmark, 6 HR/turnover |
| Classifiers | 7 | LogReg, DecisionTree, 5-NN, SVM-RBF, RandomForest, GradientBoosting, XGBoost |
| Resamplers | 4 | SMOTE, Borderline-SMOTE, ADASYN, RandomOverSampler |
| Conditions | 3 | leaked / correct / none |
| Seeds | 10 | so that every comparison carries an interval |
| Model fits | 25,620 attempted | 25,340 completed; 11,200 matched leaked–correct pairs |

Classifiers are used at library defaults except where a default is plainly unsuitable: the decision tree is capped at depth six and the forests at two hundred trees [@breiman2001]. We do not tune, because tuning would add a second experimental layer whose folds would themselves have to be made leakage-safe. The matched design therefore identifies the protocol difference conditional on these specified learners. Metrics recorded are minority-class F1, precision, recall, Matthews correlation, ROC-AUC and average precision; F1 is the headline because it is what the literature under study reports [@saito2015; @chicco2020].

Unless otherwise stated, bootstrap intervals in the descriptive summaries resample matched fits. They quantify stability within the assembled experimental record, not population-level uncertainty over a hypothetical random sample of datasets. The meta-regressions instead use standard errors clustered by dataset.

### 3.5 Controlled simulation

Real datasets confound the things we most want to separate. Highly imbalanced problems tend also to be larger, higher-dimensional and more overlapping. A synthetic grid pulls the factors apart: we generate problems at five imbalance ratios, three dimensionalities, three class-separation levels and two sample sizes, and run the leaked and correct conditions on each with five classifiers and three seeds.

### 3.6 Meta-regression

To ask what governs the premium we regress each of the three quantities—the honest score, the leaked score and their difference—on log imbalance ratio, log minority count, log feature count, log sample size, and indicator variables for classifier and resampler. Standard errors are clustered by dataset. The comparison of interest is between the $R^2$ values: if the leaked score has stopped measuring the problem, the descriptors of the problem should explain much less of its variance than they explain of the honest score’s.

### 3.7 The screening model

Each matched pair yields two labelled examples: a leaked result and an honest one. We train a gradient-boosted classifier to tell them apart using only features a reader can extract from a published paper—the reported F1, the imbalance ratio, the minority count, the feature count and the model family—and validate it leave-one-dataset-out. The construction imposes a 50:50 class prior. Its output is therefore a screening score, or a probability conditional on that experimental prior, not a calibrated posterior that an arbitrary published result is leaked.

### 3.8 Reproducibility

All datasets are public. The corpus builder downloads Group A and reads Group B from files with documented provenance, then writes a manifest that every downstream step reads. The experiment checkpoints after each dataset. All analysis flows into a single machine-readable summary, and every quantity in this manuscript is taken from that summary. Library versions at the reported run were Python 3.11.15, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0 [@pedregosa2011], imbalanced-learn 0.14.2 [@lemaitre2017] and XGBoost 3.2.0 [@chen2016]. Shared code in `leakage_lib.py` implements the three conditions so notebooks and batch runs cannot drift apart.

## 4. Results

### 4.1 The corpus behaves as intended

Table 1 and Figure 1 describe what was assembled. The imbalance ratio spans 1.02 to 109.9, minority counts 30 to 1,813, and feature counts 3 to 180. The employee-attrition datasets sit at the mild end of the imbalance range: the domain in which this error is most often committed is also the domain in which the honest score is highest, and therefore the domain in which the premium is smallest. The corpus is not a random sample of problems, and no claim here depends on treating it as one.

### 4.2 The invariance

This is the central result. Under the correct ordering, mean minority-class F1 is 0.673 with a standard deviation of 0.242. Under the leaked ordering, the mean rises to 0.896 but the standard deviation falls to 0.110. The honest score has 4.86 times the variance of the leaked one. Aggregating first to dataset means, the ratio is 4.76: across the whole corpus, mean honest F1 ranges over 0.922 while mean leaked F1 ranges over only 0.389.

**Figure 2.** The invariance. (a) Leaked against honest F1 for all 11,200 matched pairs; points lie along a band, not along a shifted diagonal. (b) Both scores against the imbalance ratio, binned. The honest score falls as the problem gets harder; the leaked score rises. (c) The premium against honest F1: how much a result is inflated is almost entirely determined by how hard the problem actually was.

As the imbalance ratio rises, mean honest F1 falls from 0.744 in the least imbalanced quartile to 0.513 in the most imbalanced. Over the same range the leaked score rises, from 0.806 to 0.958. The correlation between log imbalance ratio and F1 is $-0.268$ under the correct ordering and $0.578$ under the leaked one. Per decade of imbalance ratio the honest score falls by 0.122 while the leaked score rises by 0.120. The more imbalanced a dataset is, the more synthetic points an over-sampler must manufacture to balance it, and so the more of the test set is present in the training partition in interpolated form. Difficulty and contamination rise together, and the second more than cancels the first. This is why a high reported score on a highly imbalanced problem is the specific thing to be suspicious of.

The premium is $\mathrm{premium}=0.778-0.824\times F1(\mathrm{correct})$, with $R^2=0.795$. Nearly all of the variation in how much a result is inflated is explained by how hard the problem was to begin with.

**Table 4.** Invariance across the design factors. Each row is the mean over matched pairs at that level.

| Factor | Level | Pairs | Leaked F1 | Correct F1 | No resampling | Premium |
|---|---|---:|---:|---:|---:|---:|
| IR | 1.02–2.3 | 2,870 | 0.806 | 0.744 | — | 0.063 |
| IR | 2.33–6.0 | 2,800 | 0.869 | 0.670 | — | 0.199 |
| IR | 6.00–14.7 | 2,730 | 0.956 | 0.764 | — | 0.192 |
| IR | 14.66–109.9 | 2,800 | 0.958 | 0.514 | — | 0.445 |
| Classifier | 5NN | 1,600 | 0.883 | 0.615 | 0.598 | 0.267 |
| Classifier | DecTree | 1,600 | 0.876 | 0.638 | 0.638 | 0.238 |
| Classifier | GradBoost | 1,600 | 0.917 | 0.719 | 0.694 | 0.198 |
| Classifier | LogReg | 1,600 | 0.828 | 0.590 | 0.545 | 0.238 |
| Classifier | RandForest | 1,600 | 0.936 | 0.724 | 0.680 | 0.212 |
| Classifier | SVM-RBF | 1,600 | 0.907 | 0.686 | 0.585 | 0.221 |
| Classifier | XGBoost | 1,600 | 0.929 | 0.737 | 0.717 | 0.192 |
| Resampler | ADASYN | 2,590 | 0.904 | 0.663 | — | 0.241 |
| Resampler | Borderline | 2,870 | 0.895 | 0.674 | — | 0.221 |
| Resampler | RandomOS | 2,870 | 0.892 | 0.677 | — | 0.216 |
| Resampler | SMOTE | 2,870 | 0.895 | 0.677 | — | 0.218 |
| Domain | HR | 1,610 | 0.826 | 0.591 | — | 0.235 |
| Domain | benchmark | 9,590 | 0.908 | 0.687 | — | 0.222 |

Across imbalance quartiles the honest score spreads 0.251 and the leaked score 0.152, in opposite directions. Across classifiers the leaked score is highest for random forests and lowest for logistic regression—an ordering by capacity to memorise rather than by any property of the data. Across resamplers both scores spread by less than 0.013, so which over-sampler is used is close to irrelevant to the artefact.

The finding is not an artefact of F1 (Table 5). Matthews correlation shows a premium of 0.196 with a variance ratio of 1.60; average precision shows 0.229 and 6.91; ROC-AUC, which is insensitive to the class ratio and therefore the most conservative test available, still shows 0.056 and 1.76.

**Table 5.** The result is not specific to F1. Variance ratio is $\mathrm{Var}(\mathrm{honest})/\mathrm{Var}(\mathrm{leaked})$.

| Metric | Leaked | sd | Correct | sd | Premium | Var. ratio |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 0.896 | 0.110 | 0.673 | 0.242 | 0.224 | 4.86 |
| MCC | 0.793 | 0.220 | 0.598 | 0.278 | 0.196 | 1.60 |
| PR-AUC | 0.924 | 0.104 | 0.695 | 0.272 | 0.229 | 6.91 |
| ROC-AUC | 0.933 | 0.093 | 0.877 | 0.124 | 0.056 | 1.76 |
| Precision | 0.890 | 0.113 | 0.642 | 0.268 | 0.248 | 5.63 |
| Recall | 0.906 | 0.116 | 0.757 | 0.209 | 0.150 | 3.26 |

### 4.2b Where the F1 premium comes from

F1 conflates precision and recall. Reconstructing false negatives per 100 actual minority cases as $100(1-R)$ and false positives per 100 actual minority cases as $100R(1-P)/P$, across the corpus the leaked ordering cuts missed minority cases from 24 to 9 per 100 positives and cuts false alarms from 81 to 12 per 100 positives (Table 6). The mean F1 premium of 0.224 is the joint result of a precision lift of 0.248 and a recall lift of 0.149. Both errors fall; the larger movement is in the false-discovery rate.

That last fact is easy to misread. The leaked test set is the rebalanced, partly synthetic partition. A drop in FDR on that partition is not a drop in operational false positives on a live, imbalanced workforce. The recall lift is closer to a genuine change in the fitted decision surface; the precision lift is largely a change in the test distribution.

**Table 6.** Precision, recall and reconstructed error rates. FN and FP are per 100 actual minority cases on that protocol’s test split.

| Slice | Protocol | Precision | Recall | F1 | FN/100 pos. | FP/100 pos. |
|---|---|---:|---:|---:|---:|---:|
| All | correct | 0.642 | 0.757 | 0.673 | 24.3 | 80.9 |
| All | leaked | 0.891 | 0.907 | 0.897 | 9.3 | 11.5 |
| All | none | 0.736 | 0.594 | 0.637 | 40.6 | 14.9 |
| HR | correct | 0.584 | 0.639 | 0.591 | 36.1 | 56.5 |
| HR | leaked | 0.821 | 0.835 | 0.826 | 16.5 | 18.7 |
| Benchmark | correct | 0.652 | 0.777 | 0.687 | 22.3 | 85.0 |
| Benchmark | leaked | 0.902 | 0.919 | 0.909 | 8.1 | 10.4 |

### 4.3 What the meta-regression shows

We expected the descriptors of a problem to explain the honest score well and the leaked score poorly. They do the opposite: dataset and algorithm descriptors explain 0.565 of the variance in the leaked score and only 0.175 of the variance in the honest one (Table 7). Restricting the model to data descriptors alone, the gap is similar, 0.465 against 0.125.

The resolution is in the sign. The leaked score is highly predictable from the imbalance ratio, but it moves the wrong way. What the descriptors predict, when they predict the leaked score, is not how hard the problem is but how much contamination the pipeline manufactured. Predictability is therefore not evidence that a leaked number carries information about the problem; it carries information about the error. That is what makes the screen in Section 4.7 possible.

**Table 7.** Meta-regression of each outcome on dataset and algorithm descriptors, standard errors clustered by dataset.

| Term | Correct *b* | se | Leaked *b* | se | Premium *b* | se |
|---|---:|---:|---:|---:|---:|---:|
| const | 0.686 | 0.290 | 0.886 | 0.115 | 0.201 | 0.210 |
| log_ir | 0.488 | 0.586 | 0.802 | 0.253 | 0.314 | 0.442 |
| log_nmin | 0.711 | 0.698 | 0.792 | 0.282 | 0.081 | 0.537 |
| log_p | 0.113 | 0.074 | 0.063 | 0.022 | −0.049 | 0.058 |
| log_n | −0.705 | 0.713 | −0.798 | 0.289 | −0.093 | 0.548 |
| clf_DecTree | 0.022 | 0.024 | −0.007 | 0.008 | −0.029 | 0.019 |
| clf_GradBoost | 0.104 | 0.021 | 0.034 | 0.008 | −0.069 | 0.018 |
| clf_LogReg | −0.025 | 0.027 | −0.055 | 0.016 | −0.029 | 0.018 |
| clf_RandForest | 0.108 | 0.020 | 0.053 | 0.008 | −0.055 | 0.020 |
| clf_SVM-RBF | 0.071 | 0.014 | 0.024 | 0.009 | −0.047 | 0.008 |
| clf_XGBoost | 0.122 | 0.021 | 0.047 | 0.008 | −0.075 | 0.019 |
| sampler_Borderline | 0.007 | 0.008 | 0.000 | 0.003 | −0.006 | 0.007 |
| sampler_RandomOS | 0.010 | 0.008 | −0.002 | 0.004 | −0.012 | 0.007 |
| sampler_SMOTE | 0.010 | 0.008 | 0.000 | 0.003 | −0.009 | 0.007 |

### 4.4 The simulation isolates the cause

The corpus cannot separate imbalance from the things that travel with it. The simulation can. Across 1,350 synthetic fits, honest F1 has 8.38 times the variance of leaked F1. Varying the imbalance ratio alone, honest F1 falls from 0.773 to 0.249—a drop of 0.525—while leaked F1 moves by 0.042. Varying dimensionality alone moves the honest score by 0.153 and the leaked score by 0.026. Varying class separation moves them by 0.296 and 0.087. **Figure 3** shows all three comparisons. The premium grows with imbalance not because leakage helps more when data is scarce, but because the honest score has further to fall.

### 4.5 The mechanism, measured

The contamination rate averages 0.637 across the corpus and ranges from 0.000 to 1. Its correlation with the log imbalance ratio is 0.724. It correlates with the premium at $r=0.522$ and with the leaked score at $r=0.442$. Entering it alongside honest difficulty in a regression on the premium moves the coefficient on difficulty only from $-0.830$ to $-0.753$, about 9.3% of the total. The contamination rate measures how much of the test set has been copied into training, not what a given model does with the copies.

RandomOverSampler produces the highest contamination (0.785 against 0.464 for Borderline-SMOTE) but not the largest premium (0.216 against 0.241 for ADASYN). More contamination does not mean a proportionally larger artefact, because a duplicated point teaches a model less than an interpolated neighbourhood does. The four resamplers differ from one another by less than 0.012 in leaked F1. What matters is that synthetic points cross the split at all, not how they were made.

### 4.6 Partial resampling as a safeguard

If the artefact comes from synthetic points crossing the split, resampling to less than full balance should shrink it. At full balance the premium is 0.221. At a target minority prevalence of 0.25 it is 0.166, a reduction of 25%, and at 0.15 it is 0.155 (Table 8). The contamination rate falls in step, from 0.696 to 0.517. A quarter is not most of it. Partial resampling attenuates the artefact; it does not remove it. We are not recommending partial resampling as a way to make the error safe. The finding gives a diagnostic signature: a pipeline whose reported performance changes sharply with the resampling ratio, in a way that cannot be explained by the classifier, is a pipeline worth inspecting [@goorbergh2022; @carriero2025; @elor2022].

**Table 8.** Premium as a function of how far SMOTE is asked to rebalance the data. Intervals are bootstrap percentile intervals over matched fits.

| Target prevalence | Fits | Premium | sd | Leaked | Correct | Contamination | CI low | CI high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 162 | 0.155 | 0.180 | 0.758 | 0.603 | 0.517 | 0.128 | 0.183 |
| 0.25 | 216 | 0.166 | 0.185 | 0.814 | 0.648 | 0.601 | 0.143 | 0.190 |
| 0.40 | 261 | 0.188 | 0.199 | 0.818 | 0.630 | 0.646 | 0.165 | 0.214 |
| 0.60 | 315 | 0.201 | 0.205 | 0.843 | 0.643 | 0.672 | 0.178 | 0.223 |
| 0.80 | 333 | 0.221 | 0.210 | 0.874 | 0.653 | 0.708 | 0.198 | 0.243 |
| 1.00 | 369 | 0.221 | 0.218 | 0.877 | 0.656 | 0.696 | 0.198 | 0.244 |

**Figure 4.** (a) Premium against contamination rate, binned. (b) Premium and contamination as functions of target minority prevalence; both fall together.

### 4.7 Screening a reported result

Using only the reported F1, imbalance ratio, minority count, feature count and model family, and validating leave-one-dataset-out over 22,400 labelled results, the screen separates leaked from honest results with AUC 0.795. Accuracy at the natural threshold is 0.697 and the Brier score is 0.197. Within a dataset, per-dataset AUC has median 0.987 and exceeds 0.9 on 27 of 41 datasets. Pooled across datasets—the task a reviewer actually faces—the AUC is 0.795. Removing the reported F1 and leaving only the dataset descriptors drops the AUC to 0.500. The signal is entirely in the relationship between the reported score and the difficulty of the problem.

**Figure 5.** (a) Leave-one-dataset-out ROC. (b) Screening probability as a function of reported F1 at four imbalance ratios. The curves shift left as the problem gets harder.

SHAP values [@lundberg2017] on a 3,000-row sample confirm that mean absolute contribution is dominated by the reported F1, followed by log imbalance ratio (**Figure 6**). The screen is a classifier of pairs $(F1,\mathrm{difficulty})$: a high score on a hard problem is the leaked signature. It is not a prevalence-calibrated probability and is not evidence of error in any individual paper. Its use is to tell a reviewer when it is worth asking where the resampler sits relative to the split.

### 4.8 What the published attrition literature reports

Table 9 places our measured bands against results reported in the recent employee-attrition literature, a domain where over-sampling is near-universal and where the IBM HR benchmark gives an unusually clean comparison [@hamja2025; @nassreddine2026; @baydili2025; @alali2026; @alyousef2026; @cavescu2026]. We classify each study only by what it says about its own protocol. The two studies that state resampling was confined to the training data report minority-class F1 between 0.439 and 0.504 [@alyousef2026; @cavescu2026]. The study that states resampling preceded the split reports 0.916 [@baydili2025]. The results that do not state the ordering report between 0.659 and 0.990. Our own measurements on IBM HR put honest F1 at 0.434 and leaked F1 at 0.877.

We draw one conclusion and decline to draw another. The ordering is frequently not reported, and where it is reported the difference in outcomes is large enough that a reader cannot treat the omission as immaterial. We do not conclude that any particular unreported study is in error. Different preprocessing, splits, or a macro-averaged rather than minority-class F1 remain available explanations. That is the point: from the outside, nothing can be ruled out, which is why the ordering belongs in the methods section.

**Table 9.** Reported minority-class F1 in recent employee-attrition studies, grouped by what each paper *states* about resampling order. “Not stated” is not a ground-truth leakage label.

| Study | Dataset | Best model | Reported F1 | Stated protocol |
|---|---|---|---:|---|
| Hamja et al. (2025) | IBM HR | GradientBoosting | 0.990 | not stated |
| Nassreddine et al. (2026) | IBM HR | Ensemble (ROS) | 0.977 | not stated |
| Baydili & Tasci (2025) | IBM HR | GAN-Transformer | 0.916 | before-split |
| AL-Ali et al. (2026) | IBM HR | AdaBoost | 0.797 | not stated |
| AL-Ali et al. (2026) | Job change | HistGradientBoosting | 0.659 | not stated |
| Alyousef et al. (2026) | IBM HR | LogisticRegression | 0.504 | train-only |
| Căvescu & Popescu (2026) | IBM HR | LogisticRegression | 0.439 | train-only |

## 5. Discussion

### 5.1 Why a literature converges on numbers that look alike

Sections of the applied literature on imbalanced classification report headline scores in a narrow high band across very different datasets and model families. Read as measurements, that convergence is puzzling. Section 4.2 supplies an explanation that requires no assumption of bad faith. Under the leaked ordering the reported number is not a measurement of the problem, and quantities that do not measure the problem do not vary with it. A band is exactly what one should expect. Where a body of work reports scores that vary little across datasets of visibly different difficulty, the ordering of resampling and splitting is the first thing to look at.

### 5.2 Leakage as substitution, not inflation

“Over-optimistic” suggests a bias term: a true value plus an increment. Our results say otherwise. The leaked score correlates with the honest score at only 0.389, moves by 0.389 across a corpus over which the honest score moves by 0.922, and what movement it has runs in the wrong direction. There is little to correct, because what is there is a measurement of something else: of how much contamination the pipeline produced. If the leaked score were the honest score plus noise, the difference would be noise. Because the leaked score is close to constant, the difference is essentially the honest score with its sign flipped—which is why $R^2=0.795$.

### 5.3 What follows for review and editing

Reviewers of expert-system papers are not usually able to run the code. What they can do is ask a question whose answer is one sentence long.

1. **Require the ordering to be stated.** A methods section should say where resampling occurs relative to the split and relative to cross-validation folds.
2. **Treat a high score on a hard problem as a question, not a finding.** On a problem with an imbalance ratio near 109.9, a reported F1 that would be unremarkable at low imbalance is already unusual (Figure 5b).
3. **Ask for the no-resampling baseline.** It costs one extra run, and in 34.7% of our matched comparisons it was the stronger of the two correctly-evaluated options.

### 5.4 What follows for practice

Full rebalancing is not required by anything, and it maximises the number of synthetic points in play. Targeting a minority prevalence of 0.25 rather than 1.0 cut the premium by 25% in our corpus. That is not a licence to resample before splitting; it is an argument that the default of full balance is a choice that costs something and should be justified.

An accounting illustration on IBM HR ($n=1{,}470$, 237 leavers) makes the operational reading concrete (**Figure 7**). Suppose a one-cycle programme flags predicted leavers for a retention action costing $C_r$ per person, and replacing a leaver costs $C_t=\$15{,}000$. At $C_r=\$3{,}000$ the leaked evaluation implies a false-positive bill about \$262,000 smaller than the honest evaluation of the same models. That is the amount a team would under-provision for false positives if it treated a leaked F1 of 0.88 as an operational rate. Once the model is scored on new employees who were never in the resampling pool, the honest rates are the relevant ones.

### 5.5 Limitations

- The corpus is assembled, not sampled. Statements about the shape of the relationship are supported; statements about the typical magnitude in a given discipline are not.
- Multi-class benchmarks are binarised one-versus-rest, which may be easier than a genuinely heterogeneous minority.
- Classifiers are untuned and neural networks are not represented. Leakage-safe nested tuning and an MLP condition are needed before generalising the variance ratio.
- Datasets are capped at 5,000 rows. A within-dataset sensitivity analysis at multiple caps is needed before extrapolating to enterprise-scale data.
- The primary leaked and correct conditions do not share an identical real test set. A fixed-real-test experiment is required to isolate the training-contamination component.
- Descriptive bootstrap intervals over matched fits do not make the 11,200 comparisons independent.
- The contamination threshold is interpretable but chosen rather than estimated.
- The screen inherits the coverage of the corpus and is trained on an artificially balanced sample; numeric outputs require recalibration under unknown literature prevalence.
- We measure one leakage mechanism. Preprocessing fitted on the full dataset, feature selection before splitting, and duplicate records across the split are separate errors.

### 5.6 A reporting checklist

The following six items would, if reported, let a reader evaluate any study that applies resampling to imbalanced data. None requires extra experiments.

1. Where the resampler sits relative to the split, and relative to cross-validation folds.
2. The class balance of the test set as evaluated: real observations only, or resampled.
3. The number of minority cases before resampling, not only the ratio.
4. The score without resampling, as a baseline.
5. The resampling target prevalence, if it is not full balance.
6. Precision and recall (or TP/FP/FN counts), not F1 alone.

## 6. Conclusion

Within the evaluated capped tabular regime, resampling before splitting inflates minority-class F1 by 0.224 on average across 41 datasets, seven classifier families and four over-samplers. That number is the least interesting result in this paper. The consequential one is that the inflation is not additive: under the leaked ordering the reported score stops varying with the difficulty of the problem, retaining 4.86 times less variance than the honest score, and what variation remains runs against difficulty rather than with it. The premium is therefore predictable from difficulty alone ($R^2=0.795$), and the artefact leaves a signature stable enough to be ranked in results one did not produce (AUC 0.795 under leave-one-dataset-out validation). These conclusions concern the end-to-end erroneous workflow.

Two things follow that a practitioner of expert systems can act on. Resampling to partial rather than full balance cuts the artefact by 25% and gives a diagnostic handle on it, though it does not remove it. And a reported result can be screened from what papers already publish, which turns a general warning about leakage into a specific question about a specific study. We release the screen, the corpus builder and the full experimental record so that both can be checked and extended.

## CRediT authorship contribution statement

**Hieu Le Ngoc:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – original draft, Visualization. **Thuy Truong Dinh Hai:** Conceptualization, Validation, Writing – review & editing, Funding acquisition, Supervision.

## Acknowledgements

This research is funded by University of Finance – Marketing, HCMC Vietnam. This study is supported by Posts and Telecommunications Institute of Technology (PTIT), Vietnam. The funding bodies had no role in study design, data collection, analysis, interpretation, writing of the report, or the decision to submit the article for publication.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Cursor Grok 4.6 in order to format the source manuscript to the structure required by *Expert Systems with Applications* (title page, Highlights, numbered sections, and declarations). After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. Generative AI was not used to produce the experimental corpus, fit the models, or invent numerical results.

## Data availability

All datasets used are public. The corpus builder, the experiment, the analysis scripts and the complete record of every model fit reported here are available from <https://github.com/occbuu/leakage-premium> and are archived on Zenodo under DOI [10.5281/zenodo.22196384](https://doi.org/10.5281/zenodo.22196384).

## References
