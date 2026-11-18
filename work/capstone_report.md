
# Can March Search Performance Signals Help Identify Content at Risk of Ranking Deterioration?

**Author:** E Shirley Iren  
**Project:** FlyRank ML Internship Capstone  
**Research Lane:** Lane 1 – Ranking Signal Analysis  
**Repository:** https://github.com/irene1407/ML-1  
**Date:** November 17, 2026

---

## 0. Abstract

This project investigates whether March 2026 search-performance signals can help identify client-content pairs that experience ranking deterioration in April 2026. The analysis uses anonymized historical performance data from the FlyRank internship warehouse, with March features and April outcomes. A rule-based baseline and two machine learning models, Logistic Regression and Random Forest, were evaluated using a client-grouped train-test split. Logistic Regression achieved 0.6626 ROC-AUC, 0.8557 Average Precision, and 96% Precision@100, outperforming the baseline's 81% Precision@100. The goal is to develop a transparent, data-driven prioritization approach that helps identify content requiring further investigation while acknowledging limitations in coverage and predictive reliability.

## 1. Problem Framing

### Research question

Can March 2026 search-performance signals help identify client-content pairs whose average search ranking deteriorates in April 2026?

### Objective

The objective is to build a practical prioritization system that identifies potentially at-risk content using historical search-performance signals.

### Unit of analysis

One observation represents a client-content pair.

### Feature window

March 2026.

### Outcome window

April 2026.

### Target variable

The target, `position_worsened`, indicates whether the April impression-weighted average search position was numerically higher (worse) than the March average position.

The target is binary:

- `1`: Ranking position worsened.
- `0`: Ranking position did not worsen.

The project focuses on identifying associations and predictive patterns. It does not establish that any particular feature or action causes ranking changes.

## 2. Data Safety and Preparation

### Data source

The project uses the FlyRank internship warehouse:

https://huggingface.co/datasets/FlyRank/internship-warehouse

Dataset build: `v20260703`

The primary source table is `fact_content_daily_performance`.

### Data preparation

The development dataset was aggregated into client-content observations using March 2026 performance data.

The resulting feature frame contained:

- 158,549 observations.
- 8 columns, including identifiers, five predictive features, and the target.

### Features used

| Feature | Description |
|---|---|
| `impressions` | Number of search impressions |
| `clicks` | Number of search clicks |
| `avg_position` | Average search position |
| `march_ctr` | March click-through rate |
| `active_days` | Number of active reporting days |

Client and content identifiers were retained for grouping and traceability, not used as model features.

### Data safety measures

- Anonymized client and content identifiers were used.
- Identifiers were excluded from predictive features.
- Future outcome information was excluded from model inputs.
- Leakage checks confirmed that no forbidden or future columns were present in the approved baseline feature set.
- Private credentials and raw identifying information must not be included in public deliverables.

## 3. Baseline

A rule-based baseline was developed to identify high-visibility content with below-typical CTR for its search-position group.

### Baseline logic

1. Divide content into search-position groups.
2. Calculate the typical CTR for each group.
3. Identify high-visibility content using an impressions threshold.
4. Prioritize items whose CTR falls below the typical CTR for their position group.

The baseline score is based on:

`impressions × max(expected CTR − observed CTR, 0)`

The action assigned to qualifying content is `review_title_meta`.

### Baseline evaluation

For a fair comparison, the baseline was evaluated on the same client-grouped test set as the machine learning models. Its reference thresholds were calculated using training data only.

| Metric | Result |
|---|---:|
| Training-derived visibility threshold | 1,243 impressions |
| Selected test items | 38 |
| Precision among selected items | 86.84% |
| Recall among selected items | 0.20% |
| Precision@50 | 80% |
| Precision@100 | 81% |

The baseline demonstrates high precision among its selected items but very limited coverage.

Its Precision@K results require caution because only 38 items received non-zero scores, leaving zero-score ties that can affect ranking order.

## 4. Model and Analysis

Two machine learning models were evaluated:

- Logistic Regression
- Random Forest

### Model selection

Logistic Regression was selected for the action playbook because it achieved higher accuracy and Precision@100 than Random Forest on the grouped validation sample.

It also provides a relatively transparent modelling approach.

### Validation strategy

A client-grouped 80/20 train-test split was used.

- Random state: 42
- Training observations: 137,445
- Test observations: 21,104
- Training clients: 36
- Test clients: 10
- Client overlap: 0

Keeping clients entirely within either the training or test set helps evaluate performance on clients not seen during training.

### Feature ablation

An ablation experiment evaluated the effect of removing `avg_position`.

| Feature set | ROC-AUC | Average Precision | Precision@100 |
|---|---:|---:|---:|
| All five features | 0.6626 | 0.8557 | 0.96 |
| Without `avg_position` | 0.5896 | 0.8166 | 0.74 |

Removing `avg_position` reduced ROC-AUC by 0.0730 and Precision@100 by 22 percentage points.

This indicates that the feature contributes substantially to predictive performance. However, it is closely related to the construction of the target, so this dependency must be considered when interpreting the results.

## 5. Evaluation

### Grouped model comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 75.59% | 69.08% |
| Balanced Accuracy | 58.81% | 61.74% |
| ROC-AUC | 0.6626 | 0.6726 |
| Average Precision | 0.8557 | 0.8639 |
| Precision@100 | 96% | 94% |

Logistic Regression achieved better accuracy and Precision@100. Random Forest achieved slightly higher ROC-AUC and Average Precision.

Therefore, Logistic Regression was selected for the action playbook based on the prioritization objective, rather than being considered superior on every metric.

### Baseline versus Logistic Regression

| Metric | Baseline | Logistic Regression |
|---|---:|---:|
| Precision@50 | 80% | 94% |
| Precision@100 | 81% | 96% |

Logistic Regression improved Precision@50 by 14 percentage points and Precision@100 by 15 percentage points compared with the baseline.

These comparisons use the same grouped test observations. The baseline's zero-score ties should be considered when interpreting its Precision@K values.

## 6. Interpretation and Limitations

The results indicate that historical search-performance signals contain useful information for prioritizing content that may experience ranking deterioration.

Key findings:

1. Logistic Regression produced stronger top-ranked results than the rule-based baseline.
2. The baseline identified a small set of high-confidence candidates but had very limited recall.
3. `avg_position` contributed substantially to model performance.
4. Random Forest achieved slightly higher ROC-AUC and Average Precision, while Logistic Regression achieved higher Precision@100.

### Limitations

- The results describe one validation sample and do not guarantee future performance.
- The grouped test set contains only 10 held-out clients.
- The relationship between `avg_position` and the target may introduce target-related feature dependence.
- Precision@K measures ranking quality at selected cutoffs, not overall prediction quality.
- The baseline has limited coverage and zero-score ranking ties.
- The model identifies associations, not causal relationships.
- No claim is made that any recommended editorial action will directly improve search rankings.

## 7. Ranked Recommendations and Action Playbook

The action playbook uses model risk scores and impression volume to prioritize items for investigation.

The priority score is:

`model_risk_score × log1p(impressions)`

This score is a prioritization heuristic, not an ROI estimate or calibrated probability.

### Action categories

| Reason code | Recommended action |
|---|---|
| `HIGH_VISIBILITY_CTR_GAP` | Review title and metadata |
| `HIGH_VISIBILITY_POSITION_RISK` | Conduct an editorial refresh audit |
| `RISK_WITH_LOW_EXPOSURE` | Investigate and monitor |
| `VISIBLE_LOW_MODEL_RISK` | Protect and monitor |
| `LOW_PRIORITY_OR_LIMITED_SIGNAL` | Monitor only |

### Recommended workflow

1. Review high-priority content first.
2. Investigate CTR gaps and ranking-position risk.
3. Check content quality and search intent before making changes.
4. Monitor subsequent performance.
5. Re-evaluate recommendations using future observations.

The playbook should support human decision-making rather than automatically changing content.

## 8. Reproducibility

The project uses Python-based data preparation, feature engineering, model training, validation, and evaluation.

### Main tools and libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

### Project notebooks

The analysis is organized into separate notebooks covering:

- Research question and problem framing
- Data contract and feature preparation
- Feature leakage checks
- Baseline scoring
- Signal audit
- Machine learning model development
- Validation audit
- Action playbook

### Repository

https://github.com/irene1407/ML-1

The notebooks and supporting documentation provide the project workflow and evaluation methodology.

## 9. Acknowledgments and Data Credit

This project was completed as part of the FlyRank ML Internship Capstone.

Dataset and project credit:

https://flyrank.ai

The analysis uses the provided anonymized internship warehouse data and is intended for educational and research purposes.
