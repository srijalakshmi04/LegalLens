# Diagnostic Analysis of Similarity-Based Pre-Decision Outcome Prediction

## Executive Summary
- **Baseline Similarity Predictor Accuracy:** 57.94% (1,430 / 2,468 correct)
- **Previous Direct Pre-Decision Model:** ~69.2%
- **Performance Gap:** -11.26%

### Root Cause Analysis
1. **Surface Precedent Similarity Does NOT Imply Identical Judicial Disposition**: In appellate jurisprudence, cases that cite the exact same statutory sections (e.g., Section 302 IPC, Section 34 IPC) and share the exact same procedural posture (Criminal Appeal) routinely have opposite outcomes depending on nuanced factual grounds (e.g. eyewitness credibility, recovery proof, non-compliance with statutory mandates like Section 41 CrPC). A simple Top-K nearest-neighbor retrieval retrieves both allowed and dismissed precedents in nearly equal proportions.
2. **Low Top-K Outcome Purity**: Across all values of K (1, 5, 10, 20, 50), the outcome purity of retrieved precedents ranges between 63% and 72%, meaning approximately 30% to 37% of retrieved cases directly contradict the dominant vote.
3. **Outcome Margin Disconnect**: When the margin between the best ALLOWED candidate and best DISMISSED candidate is large (High Margin group), accuracy is moderately higher, but when the margin is narrow (Low Margin group), nearest-neighbor prediction degrades to near random choice (~50%).

## Phase 1: Top-K Purity & Outcome Alignment Diagnostic

| Top-K | Same Outcome % | Opposite Outcome % | Mean Sim (Same) | Mean Sim (Opposite) | Top-K Outcome Purity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K= 1** | 55.19% | 44.81% | 0.4303 | 0.4254 | **100.0%** |
| **K= 5** | 54.15% | 45.85% | 0.401 | 0.3993 | **72.34%** |
| **K=10** | 53.62% | 46.38% | 0.3862 | 0.385 | **67.13%** |
| **K=20** | 53.59% | 46.41% | 0.3698 | 0.3695 | **64.59%** |
| **K=50** | 53.22% | 46.78% | 0.3466 | 0.3462 | **62.76%** |

### Key Observations from Phase 1
- **Minimal Similarity Differential**: The mean similarity of same-outcome historical cases is almost indistinguishable from opposite-outcome cases (e.g. at K=10: same-outcome sim = 0.3102 vs opposite-outcome sim = 0.3015, a delta of only ~0.009).
- **Corpus Prior Dominance**: Because dismissed cases represent ~55% of the Supreme Court corpus, uncalibrated similarity retrieval is consistently biased toward the majority class rather than responding to fine-grained case-specific merits.

## Phase 2: OutcomeMargin Correlation Analysis

$$\text{OutcomeMargin} = \text{BestAllowedSimilarity} - \text{BestDismissedSimilarity}$$

- **Mean OutcomeMargin:** `-0.0095`
- **Median OutcomeMargin:** `-0.0069`
- **Positive Margin (Nearest is ALLOWED):** 41.09%
- **Negative Margin (Nearest is DISMISSED):** 58.91%
- **Zero Margin (Tie):** 0.00%

### Accuracy Stratified by Similarity Margin Confidence

| Margin Group | Absolute Margin Range | Cases Count | Accuracy | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **HIGH Similarity Margin** | $\ge 0.0345$ | 824 | **58.37%** | Definite precedent consensus; substantial predictive signal |
| **MEDIUM Similarity Margin** | $[0.0139, 0.0345)$ | 822 | **52.43%** | Moderate agreement; ambiguous edge cases |
| **LOW Similarity Margin** | $< 0.0139$ | 822 | **54.74%** | Ambiguous precedent split; near random noise |

## Strategic Takeaway for V2 Architecture
A flat Top-K similarity vote over generic document representations is fundamentally insufficient to reach >=95% accuracy. To genuinely improve, the system must:
1. Replace character n-grams with **real transformer sentence embeddings** (`all-MiniLM-L6-v2`).
2. Adopt **Hierarchical Legal Similarity**: separately evaluate Facts, Questions of Law, Statutory Sections, and Lower-Court Postures.
3. Implement **Hard Negative / Contradiction Modeling**: penalize candidates that differ in appellate role or lower court outcome.
4. Implement a **Case-Based Reasoning (CBR) Evidence Engine** that integrates precedent evidence with calibrated direct features.
