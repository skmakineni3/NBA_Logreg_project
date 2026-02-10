# NBA Win Prediction with Logistic Regression

Predict NBA game outcomes (win/loss) using logistic regression models built from game-level box score statistics. This project starts with a simple baseline model using points scored (PTS) and then extends to a multivariate model using multiple box-score features. The final model is evaluated with a stratified train/test split.

## Project Overview

**Goal:** Fit logistic regression models to estimate  
\[
\hat{y} = P(\text{Win}=1 \mid X)
\]
where `Win=1` indicates the team won the game.

**Why logistic regression?** It produces calibrated win probabilities and a natural decision rule via thresholding (e.g., predict win if \(\hat{y} \ge 0.5\)).

## Key Ideas and Modeling Choices

- **Baseline model:** Predict win probability from **PTS only**.
- **Intercept (bias):** Included to allow the decision boundary to shift (more realistic than a no-intercept curve).
- **Loss functions:** Compared **MSE** vs. **cross-entropy (log loss)** and visualized loss surfaces to understand optimization behavior.
- **Leakage control:** Excluded `PLUS_MINUS` since it effectively encodes the outcome via point differential (would make results near-perfect but not meaningful for prediction).
- **Multicollinearity note:** Box-score features (e.g., PTS, FGM, FGA, FG%) are strongly correlated, so individual coefficients may appear counterintuitive. Coefficients represent conditional effects and should be interpreted cautiously without regularization/feature selection.

## Results (Stratified 80/20 Train/Test Split)

### Baseline: PTS-only Logistic Regression
- **Test Accuracy:** 0.6992  
- **ROC-AUC:** 0.7550  
- **Log Loss:** 0.5952  
- **Confusion Matrix:** \(\begin{bmatrix}177 & 69\\79 & 167\end{bmatrix}\)

### Full Model: Box-Score Features (FGM:PTS), excluding `PLUS_MINUS`
- **Test Accuracy:** 0.8577  
- **ROC-AUC:** 0.9298  

### Improvement Over Baseline
- Accuracy improved from **0.6992 → 0.8577** (**+15.85 percentage points**)  
- ROC-AUC improved from **0.7550 → 0.9298** (**+0.1748**)

## Repository Structure (suggested)


